ret = ""
response = f"<think>\n{ret['choices'][0]['message']['reasoning_content']}\n</think>\n{ret['choices'][0]['message']['content']}" if ret else ""