import re

from AgentFactory.Models.LLM import LLM_local

class DeepSeek(LLM_local):

    def __init__(self, model_name, device = "cuda", bf16 = False):
        
        super().__init__(model_name, device, bf16)

    # def format_prompt(self, msgs, assist_prefix:str=""):
    #     """ Deepseek output format:
    #         "{system prompt}<｜User｜>hello<｜Assistant｜>hello<｜end▁of▁sentence｜>"
    #     Note1: In the paper: A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.<｜User｜>{user prompt}<｜Assistant｜<think>...</think> <answer>...</answer>｜end▁of▁sentence｜>
    #     Note2: <｜begin▁of▁sentence｜> is the BOS token, which will processed automatically by the tokenizer
    #     """
        
    #     user_msg_w_sys_template = """{system_msg}<｜User｜>{user_msg}"""

    #     user_msg_template = """<｜User｜>{user_msg}"""

    #     assistant_msg_template = """<｜Assistant｜>{assistant_msg}"""

    #     if msgs[0]["role"] == "system":
    #         prompt = user_msg_w_sys_template.format(system_msg=msgs[0]["content"], user_msg=msgs[1]["content"])
    #         start_idx = 2
    #     elif msgs[0]["role"] == "user":
    #         prompt = user_msg_template.format(user_msg=msgs[0]["content"])
    #         start_idx = 1

    #     for msg in msgs[start_idx:]:
    #         if msg["role"] == "user":
    #             prompt += user_msg_template.format(user_msg=msg["content"])
    #         elif msg["role"] == "assistant":
    #             prompt += assistant_msg_template.format(assistant_msg=msg["content"])

    #     prompt += f"<｜end▁of▁sentence｜>"

    #     # due to thinking, assist_prefix may not available in DeepSeek
    #     if msgs[-1]["role"] == "user":
    #         prompt += f"<｜Assistant｜>{assist_prefix}"

    #     return prompt

    def extract_response(self, output, assist_prefix:str=""):
        """ Deepseek output format:
            "{system prompt}<｜User｜>hello<｜Assistant｜>hello<｜end▁of▁sentence｜>"
        """
        #extract last model replay between "<｜Assistant｜>" and "<｜end▁of▁sentence｜>"
        pattern = r"<\｜Assistant\｜>(.+?)<\｜end▁of▁sentence\｜>"
        matches = re.findall(pattern, output, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None