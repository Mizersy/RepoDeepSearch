# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a simple multi-turn dataset for testing
"""

import argparse
import os
import json
import pandas as pd
import re

def load_loc_file(file_path):
    with open(file_path, "r") as f:
        loc_file_list = [json.loads(line) for line in f]
    loc_file_messages = []
    for item in loc_file_list:
        for traj in item["file_loc_trajs"]:
            message = traj["prompt"]
            message.append({
                "role": "assistant",
                "content": traj["response"]
            })
            loc_file_messages.append({
                "messages": message
            })
    return loc_file_messages

def load_loc_func(file_path):
    with open(file_path, "r") as f:
        loc_func_list = [json.loads(line) for line in f]
    loc_func_messages = []
    for item in loc_func_list:
        # for traj in item["func_loc_trajs"]:
        traj = item["func_loc_trajs"][-1]
        message = traj["prompt"]
        message.append({
            "role": "assistant",
            "content": traj["response"]
        })
        loc_func_messages.append({
            "messages": message
        })
    return loc_func_messages


def process_message_for_native_function_call(message):
    for i in range(len(message)):
        if message[i]["role"] == "system":
            function_call_prompt = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_code_of_class\", \"description\": \"A tool for geting the class content of specific file\", \"parameters\": {\"type\": \"object\", \"properties\": {\"file_name\": {\"type\": \"string\", \"description\": \"The full file name\", \"enum\": null}, \"class_name\": {\"type\": \"string\", \"description\": \"class name\", \"enum\": null}}, \"required\": [\"file_name\", \"class_name\"]}, \"strict\": false}}\n{\"type\": \"function\", \"function\": {\"name\": \"get_code_of_class_function\", \"description\": \"A tool for geting the function content of specific class\", \"parameters\": {\"type\": \"object\", \"properties\": {\"file_name\": {\"type\": \"string\", \"description\": \"The full file name\", \"enum\": null}, \"class_name\": {\"type\": \"string\", \"description\": \"class name\", \"enum\": null}, \"func_name\": {\"type\": \"string\", \"description\": \"function name\", \"enum\": null}}, \"required\": [\"file_name\", \"class_name\", \"func_name\"]}, \"strict\": false}}\n{\"type\": \"function\", \"function\": {\"name\": \"get_code_of_file_function\", \"description\": \"A tool for geting the function content of specific file\", \"parameters\": {\"type\": \"object\", \"properties\": {\"file_name\": {\"type\": \"string\", \"description\": \"The full file name\", \"enum\": null}, \"func_name\": {\"type\": \"string\", \"description\": \"function name\", \"enum\": null}}, \"required\": [\"file_name\", \"func_name\"]}, \"strict\": false}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            message[i]["content"] = message[i]["content"].split("Function calls you can use are as follows:")[0]
            message[i]["content"] = message[i]["content"] + function_call_prompt
        elif message[i]["role"] == "user":
            # print(message[i]["content"])
            if "Your response should follow the format as follows:" in message[i]['content']:
                message[i]['content'] = message[i]['content'].split("Your response should follow the format as follows:")[0].replace("<action>","<tool_call>").replace("</action>","</tool_call>")
            elif "Here is a result of a function/class code retrived by" in message[i]['content']:
                
                def extract_function_call(content):
                    import re
                    matches = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
                    if matches:
                        return matches[0]
                    else:
                        return content
                function_call_content = extract_function_call(message[i-1]['content'])
                message[i]['content'] = f"Here is a result of a function/class code retrived by \"{function_call_content}\"\n<code>" + message[i]['content'].split("<code>")[1]
            elif "Please call functions in the right format to get enough information for your final answer" in message[i]['content']:
                message[i]['content'] = """wrong tool call format! For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
For example:
<tool_call>
{"name": "get_code_of_class", "arguments": {    "file_name": "file1.py",    "class_name": "class1"  }}
</tool_call>"""
            elif "Based on the available information, reconfirm and provide complete name of the top-5 most likely culprit locations for the bug." in message[i]['content']:
                continue
            else:
                print("new user content!")
                breakpoint()
            
        elif message[i]["role"] == "assistant":
            def filter_function_call(content):
                import re
                matches = re.findall(r'<action>\n```(.*?)```\n</action>', content, re.DOTALL)
                if matches:
                    return matches[0]
                else:
                    return content
                # return content.replace("```python", "").replace("`", "").replace("'", "").replace('"', '')
            try:
                function_call = filter_function_call(message[i]["content"])
                function_name = function_call[:function_call.find('(')].strip()
                arguments = function_call[function_call.find('(') + 1:function_call.rfind(')')].strip()
                args = [arg.strip() for arg in re.split(r",\s*(?![^()]*\))", arguments)]
                # breakpoint()
                # 确保参数数量不超过三个
                arg1, arg2, arg3 = (args + ['None'] * 3)[:3]
                if function_name == 'get_functions_of_class':
                    arg1 = arg1.replace("'", "").replace('"', "")
                    # function_retval = get_functions_of_class(arg1, self.instance_id)
                    
                elif function_name == 'get_code_of_class':
                    arg1 = arg1.replace("'", "").replace('"', "")
                    arg2 = arg2.replace("'", "").replace('"', "")
                    # function_retval = get_code_of_class(arg1, arg2, self.instance_id)
                    function_call_content = f"<tool_call>\n{{\"name\": \"get_code_of_class\", \"arguments\": {{    \"file_name\": \"{arg1}\",    \"class_name\": \"{arg2}\"  }} }}\n</tool_call>"
                elif function_name == 'get_code_of_class_function':
                    arg1 = arg1.replace("'", "").replace('"', "")
                    arg2 = arg2.replace("'", "").replace('"', "")
                    arg3 = arg3.replace("'", "").replace('"', "")
                    # function_retval = get_code_of_class_function(arg1, arg2, arg3, self.instance_id)
                    function_call_content = f"<tool_call>\n{{\"name\": \"get_code_of_class_function\", \"arguments\": {{    \"file_name\": \"{arg1}\",    \"class_name\": \"{arg2}\",    \"func_name\": \"{arg3}\"  }} }}\n</tool_call>"
                elif function_name == 'get_code_of_file_function':
                    arg1 = arg1.replace("'", "").replace('"', "")
                    arg2 = arg2.replace("'", "").replace('"', "")
                    # function_retval = get_code_of_file_function(arg1, arg2, self.instance_id)
                    function_call_content = f"<tool_call>\n{{\"name\": \"get_code_of_file_function\", \"arguments\": {{    \"file_name\": \"{arg1}\",    \"func_name\": \"{arg2}\"  }} }}\n</tool_call>"
                else:
                    continue
                
                message[i]["content"] = message[i]["content"].split("<action>")[0] + function_call_content
            except Exception as e:
                print(e)
                breakpoint()
            # breakpoint()
    return message

def load_loc_messages(file_path):
    with open(file_path, "r") as f:
        loc_list = [json.loads(line) for line in f]
    print(len(loc_list))
    messages = []
    for item in loc_list:
        if "file_loc_trajs" in item:
            for traj in item["file_loc_trajs"]:
                message = traj["prompt"]
                message.append({
                    "role": "assistant",
                    "content": traj["response"]
                })
                messages.append({
                    "messages": message
                })
        else:
            
            try:
                if "func_loc_trajs" in item:
                    traj = item["func_loc_trajs"][-1]
                else:
                    traj = item['func_traj']
            except:
                print(item.keys())
                breakpoint()
            message = traj["prompt"]
            message.append({
                "role": "assistant",
                "content": traj["response"]
            })
            message = process_message_for_native_function_call(message)
            messages.append({
                "messages": message
            })
    return messages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_files", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    conversations = []
    for loc_file in args.loc_files.split(","):
        conversations.extend(load_loc_messages(loc_file))
    

    # Shuffle conversations
    import random
    random.shuffle(conversations)

    train_split_ratio = 0.95
    train_size = int(len(conversations) * train_split_ratio)
    train_data = conversations[:train_size]
    test_data = conversations[train_size:]

    # Create output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save to parquet files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"))
    metadata = {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_split_ratio": train_split_ratio,
        "loc_files": args.loc_files,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # Print statistics
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {output_dir}")


if __name__ == "__main__":
    main()
