import regex as re

def format_llama2(msgs:list, assist_prefix:str=""):
    """
    llama2-chat input fomat:
        "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]"
    msgs: [
        {
            "system_msg": "Hi there!"
        },
        {
            "user_msg": "Hello",
        },
        {
            "assistant_msg": "How are you?",
        },
        {
            "user_msg": "I'm good",
        },
        ...
    ]
    """

    user_msg_w_sys_template = """<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST] """
    user_msg_template       = """<s>[INST] {user_msg} [/INST] """
    assistant_msg_template  = """{assistant_msg} </s>"""
    prompt = user_msg_w_sys_template.format(system_msg=msgs[0]["system_msg"], user_msg=msgs[1]["user_msg"])
    for msg in msgs[2:]:
        if "user_msg" in msg:
            prompt += user_msg_template.format(user_msg=msg["user_msg"])
        elif "assistant_msg" in msg:
            prompt += assistant_msg_template.format(assistant_msg=msg["assistant_msg"])
    
    if "user_msg" in msgs[-1]:
        prompt += f"{assist_prefix}"
         
    return prompt

def extract_llama2(output:str, assist_prefix:str=""):
    """
    llama2-chat output format:
        "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST] {model_reply}</s><s>[INST] {user_message_2} [/INST] {model_reply}</s>"
    Note:
        </s> will be ignored by the model
    """
    #extract last model replay
    return output.split(f"[/INST] {assist_prefix}")[-1].split("[/INST]")[-1].strip()

def format_llama3(msgs:list, assist_prefix:str=""):
    """
    llama3-chat output format:
        "<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assis_msg}<|eot_id|>"
    Note:
    
    """
    user_msg_w_sys_template = """<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"""

    user_msg_template = """<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"""

    assistant_msg_template = """<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>"""

    prompt = user_msg_w_sys_template.format(system_msg=msgs[0]["system_msg"], user_msg=msgs[1]["user_msg"])
    
    for msg in msgs[2:]:
        if "user_msg" in msg:
            prompt += user_msg_template.format(user_msg=msg["user_msg"])
        elif "assistant_msg" in msg:
            prompt += assistant_msg_template.format(assistant_msg=msg["assistant_msg"])

    if "user_msg" in msgs[-1]:
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assist_prefix}"
         
    return prompt

def extract_llama3(output:str, assist_prefix:str=""):
    """
    llama3-chat output format:
        "<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assis_msg}<|eot_id|>"
    Note:
    
    """
    #extract last model replay between "<|start_header_id|>assistant<|end_header_id|>" and "<|eot_id|>"
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.+?)<\|eot_id\|>"
    matches = re.findall(pattern, output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def format_blank(msgs:list, assist_prefix:str=""):
    """
    blank output format:
        {sys_msg}
        {user_msg}
        {assis_msg}
    Note:
    
    """
    user_msg_w_sys_template = """{system_msg}\n{user_msg}\n"""

    user_msg_template = """{user_msg}\n"""

    assistant_msg_template = """{assistant_msg}\n"""

    prompt = user_msg_w_sys_template.format(system_msg=msgs[0]["system_msg"], user_msg=msgs[1]["user_msg"])
    
    for msg in msgs[2:]:
        if "user_msg" in msg:
            prompt += user_msg_template.format(user_msg=msg["user_msg"])
        elif "assistant_msg" in msg:
            prompt += assistant_msg_template.format(assistant_msg=msg["assistant_msg"])

    if assist_prefix:
        prompt += f"{assist_prefix}"
         
    return prompt

def extract_blank(output:str, assist_prefix:str=""):
    """
    blank output format:
        "{sys_msg}
        {user_msg}
        {assis_msg}"
    Note:
    
    """
    # return last line
    return output.strip.split("\n")[-1]

model2format_func = {
    "NousResearch/Llama-2-7b-hf": 
    {
        "encode": format_llama2,
        "decode": extract_llama2,
    },
    "meta-llama/Meta-Llama-3-8B":
    {
        "encode": format_llama3,
        "decode": extract_llama3,
    },
    "meta-llama/Meta-Llama-3-8B-Instruct":
    {
        "encode": format_llama3,
        "decode": extract_llama3,
    },
    # "meta-llama/Meta-Llama-3-8B":
    # {
    #     "encode": format_blank,
    #     "decode": extract_blank,
    # }
}
