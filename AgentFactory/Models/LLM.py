# TBD: LLM_local / LLM_remote

import threading
import torch
import copy

from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod

class LLM(ABC):
    def __init__(self, model_name):
        
        self.model_name = model_name
        
        self.lock = threading.Lock()

        self._roleNames = {
            "system": "system",
            "user": "user",
            "assistant": "assistant"
        }
    
    @abstractmethod
    def get_response(self):
        pass

    @property
    def roleNames(self):
        return self._roleNames
    
    @roleNames.setter
    def roleNames(self, roleNames):
        self._roleNames = roleNames

    def __str__(self):
        return f'LLM(model={self.model_name})'

    def __repr__(self):
        return str(self)
    
class LLM_local(LLM):
    
    def __init__(self, model_name, device = "cuda", bf16 = False):

        super().__init__(model_name)
        
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if bf16:
            self.model = self.model.bfloat16()
        self.model.to(self.device)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def format_prompt(self, msg_history, assist_prefix = ""):
        """
        msgs:
            [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
                {"role": "user", "content": "I'd like to show off how chat templating works!"},
            ]
        """
        
        def single_modalitize(_msgs):
            msgs = copy.deepcopy(_msgs)
            for msg in msgs:
                if type(msg["content"]) == list:
                # concatenate the list of strings in msg["content"] into a single string
                    msg["content"] = "".join([it["text"] for it in msg["content"]])
            return msgs
        
        msgs = single_modalitize(msg_history)
        formatted_prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        formatted_prompt += f"{assist_prefix}"

        return formatted_prompt

    @abstractmethod
    def extract_response(self, output):
        pass

    def get_response(self, msg_history, images = [], max_length = 2000,
                     do_sample = True, num_beams = 1,
                     top_k = 50, top_p = 0.95, temperature = 1.0, reply_prefix = ""):

        formatted_prompt = self.format_prompt(msg_history, reply_prefix)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with self.lock:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(**inputs, 
                        do_sample = do_sample, max_length=max_length, 
                        num_beams = num_beams, top_k = top_k, top_p = top_p, temperature = temperature,
                        pad_token_id=self.tokenizer.eos_token_id)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        texts = self.extract_response(response)

        return response, texts

    def train(self):
        pass

    def save(self, output_dir):
        
        if hasattr(self.model, "module"):
            self.model.module.config.save_pretrained(output_dir)
            self.model.module.save_pretrained(output_dir)
        else:
            self.model.config.save_pretrained(output_dir)
            self.model.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(f'{output_dir}/{self.model_name}')

class LLM_remote(LLM):

    def __init__(self, model_name, api_key):

        super().__init__(model_name)
        
        self.api_key = api_key
    
    @abstractmethod
    def get_response(self):
        pass

class MLLM_local(LLM):
    
    def __init__(self, model_name, device = "cuda", bf16 = False):

        super().__init__(model_name)
        
        self.device = device
        self.bf16 = bf16

    @abstractmethod
    def get_response(self):
        pass

class MLLM_remote(LLM):
    
    def __init__(self, model_name, api_key):

        super().__init__(model_name)

        self.api_key = api_key

        self.input_tokens = 0
        self.output_tokens = 0

    @abstractmethod
    def get_response(self):
        pass

    @abstractmethod
    def total_cost(self):
        pass