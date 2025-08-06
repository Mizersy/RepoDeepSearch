import time
from typing import Dict, Union

import openai
import anthropic
import tiktoken
from openai import BadRequestError, OpenAI
import re
import os
import json
import requests
from typing import Any, Dict, List, Mapping, Optional, ClassVar, Callable, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

def get_jwt_token(model):
    jwt_url = '<url>'
    if model == "aws_claude35_sdk_sonnet_v2":
        secret = "<api_key>" # claude3.5
    elif model == "aws_sdk_claude37_sonnet": 
        secret = "<api_key>" # claude3.7
    else:
        raise Exception(f"Unknown model: {model}")
    #print(secret)
    headers = {'Authorization': f'Bearer {secret}'}
    response = requests.get(jwt_url, headers=headers)
    return response.headers["X-Jwt-Token"]

def get_extra_header(model):
    token = get_jwt_token(model)
    headers = {"X-JWT-TOKEN": token}
    return headers

def send_request_devgpt(model, messages, temperature = 0.0, n = 1, max_tokens = 1024):
    data = {}
    request_args = {
        'disable_antidirt': True,
        'model': model,
        'request_timeout': None,
        'max_tokens': max_tokens,
        'stream': False,
        'n': n,
        'temperature': temperature,
        'top_p': 1,
        'api_key': None,
        'api_base': None,
        'organization': None,
        'stop_sequences': ["</action>"]
    }
    data["messages"] = messages
    length = sum(len(str(obj)) for obj in messages)
    data.update(request_args)

    session = requests.Session()

    retry_limit = 3
    current_retry = 0
    while True:
        retry = Retry(
            total=15, 
            backoff_factor=2,  
            status_forcelist=tuple(range(401, 6000)),  
            allowed_methods=["POST"] 
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        chat_url = "<url>"

        response = session.post(chat_url, json=data, headers=get_extra_header(model), timeout=(120, 120))
        resp_json = response.json()

        if resp_json["code"] != 0 or response.status_code != 200:
            print("response.status_code: ", response.status_code)
            if length > 200000:
                raise Exception(f"Token limit protect! Message length: {length}\n{resp_json}")
            time.sleep(30) #防止tpm被打满
            current_retry += 1
            if current_retry >= retry_limit:
                break
            else:
                continue
            
        resp_json = response.json()
        return resp_json["data"]

    raise Exception(f"Failed to get response from LLM, Message length: {length}\n{resp_json}")

def send_request_openapi(model, messages, temperature = 0.0, n = 1, max_tokens = 1024):
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 8192,
        'temperature': 0.6,
        'top_p': 0.95,
        "stream": False,
        'stop_sequences': ["</action>"],
    }
    length = sum(len(str(obj)) for obj in messages)

    session = requests.Session()

    retry_limit = 3
    current_retry = 0
    while True:
        retry = Retry(
            total=15, 
            backoff_factor=2,  
            status_forcelist=[429, 500, 502, 503, 504],  
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        chat_url = "<url>"

        response = session.post(chat_url, json=data, headers=headers, timeout=(120, 600))

        parse_error = False
        resp_json = {}
        try:
            resp_json = response.json()
            _ = resp_json["choices"][0]
        except:
            parse_error = True

        if parse_error or response.status_code != 200 or "error" in resp_json["choices"][0]:
            print("response.status_code: ", response.status_code)
            # print(response.text)
            if length > 200000:
                raise Exception(f"Token limit protect! Message length: {length}")
            time.sleep(30)  # 防止tpm被打满
            current_retry += 1
            if current_retry >= retry_limit:
                break
            else:
                continue

        return resp_json

    raise Exception(f"Failed to get response from LLM, Message length: {length}")


# def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
#     """Returns the number of tokens used by a list of messages."""
#     try:
#         encoding = tiktoken.encoding_for_model(model)
#     except KeyError:
#         encoding = tiktoken.get_encoding("cl100k_base")
#     if isinstance(message, list):
#         # use last message.
#         num_tokens = len(encoding.encode(message[0]["content"]))
#     else:
#         num_tokens = len(encoding.encode(message))
#     return num_tokens

def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = 0
        for msg in message:
            num_tokens += len(encoding.encode(msg["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a debugging assistant of our Python software.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        if message[0]['role'] == 'system':
            config = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": batch_size,
                "messages": message,
            }
        else:
            config = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": batch_size,
                "messages": [{"role": "system", "content": system_message}] + message,
            }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    # openai_api_base = "0.0.0.0:8000/v1"

    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # if "deepseek" in config["model"]:
    #     client = openai.OpenAI(api_key="", base_url="https://api.deepseek.com/v1")
    # elif config["model"] == "llama3":
    #     client = openai.OpenAI(api_key="token-abc123", base_url="http://127.0.0.1:7333/v1")
    # else:
    #     # client = openai.OpenAI(api_key="", base_url="")
    #     client = openai.OpenAI(api_key="",
    #                            base_url="https://api.siliconflow.cn/v1")


    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                if e.code == 400:
                    return None
                # raise Exception("Invalid API Request")
                return None
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(50)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(50)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(50)

        retries += 1

    logger.info(f"API response {ret}")
    return ret

def chat_vllm_server(model_name, messages, temperature, max_tokens=4096, top_p=1):

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    # openai_api_base = "0.0.0.0:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    try:
        chat_response = client.chat.completions.create(
            model=model_name,  # Lingma-SWE-GPT, Lingma-SWE-GPT-small
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # extra_body={
            # "repetition_penalty": 1.05,
            # },
            # skip_special_tokens=False,
            # stop=["<|im_end|>"]
        )
    except Exception as e:
        print(e)
        chat_response = None
    
    return chat_response  


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False
):
    ret = None
    retries = 0

    client = anthropic.Anthropic(api_key="", base_url="")


    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10 * retries)
        retries += 1
    logger.info(ret)

    return ret

# def request_anthropic_engine(
#     config, logger, max_retries=40, timeout=500, prompt_cache=False
# ):
#     ret = None
#     retries = 0
#
#     client = openai.OpenAI(api_key="", base_url="")
#
#     while ret is None and retries < max_retries:
#         try:
#             # Attempt to get the completion
#             logger.info("Creating API request")
#
#             ret = client.chat.completions.create(**config)
#
#         except openai.OpenAIError as e:
#             if isinstance(e, openai.BadRequestError):
#                 logger.info("Request invalid")
#                 print(e)
#                 logger.info(e)
#                 raise Exception("Invalid API Request")
#             elif isinstance(e, openai.RateLimitError):
#                 print("Rate limit exceeded. Waiting...")
#                 logger.info("Rate limit exceeded. Waiting...")
#                 print(e)
#                 logger.info(e)
#                 time.sleep(10 * retries)
#             elif isinstance(e, openai.APIConnectionError):
#                 print("API connection error. Waiting...")
#                 logger.info("API connection error. Waiting...")
#                 print(e)
#                 logger.info(e)
#                 time.sleep(10 * retries)
#             else:
#                 print("Unknown error. Waiting...")
#                 logger.info("Unknown error. Waiting...")
#                 print(e)
#                 logger.info(e)
#                 time.sleep(10 * retries)
#
#         retries += 1
#
#     logger.info(f"API response {ret}")
#     return ret