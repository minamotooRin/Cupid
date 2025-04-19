import re

from AgentFactory.Models.LLM import LLM_local

class LLaMA3(LLM_local):
    
    def __init__(self, model_name, device = "cuda", bf16 = False):
        
        super().__init__(model_name, device, bf16)
    
    # def format_prompt(self, msgs, assist_prefix:str=""):
    #     """
    #     llama3-chat output format:
    #         "<|start_header_id|>system<|end_header_id|>\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{assis_msg}<|eot_id|>"
    #     Note:
    
    #     """
    #     user_msg_w_sys_template = """<|start_header_id|>system<|end_header_id|>\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>"""

    #     user_msg_template = """<|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>"""

    #     assistant_msg_template = """<|start_header_id|>assistant<|end_header_id|>\n{assistant_msg}<|eot_id|>"""

    #     prompt = user_msg_w_sys_template.format(system_msg=msgs[0]["content"], user_msg=msgs[1]["content"])
        
    #     for msg in msgs[2:]:
    #         if msg["role"] == "user":
    #             prompt += user_msg_template.format(user_msg=msg["content"])
    #         elif msg["role"] == "assistant":
    #             prompt += assistant_msg_template.format(assistant_msg=msg["content"])

    #     if msgs[-1]["role"] == "user":
    #         prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assist_prefix}"
            
    #     return prompt
    
    def extract_response(self, output):
        """
        llama3-chat output format:
            "<|start_header_id|>system<|end_header_id|>\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{assis_msg}<|eot_id|>"
        Note:
        
        """
        #extract last model replay between "<|start_header_id|>assistant<|end_header_id|>" and "<|eot_id|>"
        pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n(.+?)<\|eot_id\|>"
        matches = re.findall(pattern, output, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None