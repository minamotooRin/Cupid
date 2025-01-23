import torch
import re
import random
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from repe import repe_pipeline_registry, WrappedReadingVecModel
from abc import ABC, abstractmethod

repe_pipeline_registry()

class ModelPool:
    def __init__(self):

        self.type2class = {
            "LLaMA2": LLaMA2,
            "LLaMA3": LLaMA3,
        }

        self.models = {}
    
    def __getitem__(self, model_name):
        return self.models[model_name]

    def add_model(self, model_type, model_name, device, bf16 = False):
        if model_name in self.models:
            return
        self.models[model_name] = self.type2class[model_type](model_name, device, bf16)


class LLM(ABC):
    def __init__(self, model_name, device = "cuda", bf16 = False):
        
        self.model_name = model_name
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        if bf16:
            self.model = self.model.bfloat16()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # The WrappedReadingVecModel will affect the model's forward pass, so it should be used in a thread-safe way
        self.wrapped_model = WrappedReadingVecModel(self.model, self.tokenizer)
        self.lock = threading.Lock()

    @abstractmethod
    def format_prompt(self, prompt, assist_prefix):
        pass

    @abstractmethod
    def extract_response(self, output, assist_prefix):
        pass
    
    def get_controller_activations(self, pn_pairs, layers_id, coeff, batch_size):
        rep_token = -1 # the last token is used for representation
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        n_difference = 1
        direction_method = 'pca'
        train_labels = []
        for d in pn_pairs:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
            
        with self.lock:
            rep_reading_pipeline = pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
            rep_reading_pipeline.tokenizer.pad_token_id = self.model.config.eos_token_id

            rep_reader = rep_reading_pipeline.get_directions(
                pn_pairs, 
                rep_token=rep_token, 
                hidden_layers=hidden_layers, 
                n_difference=n_difference, 
                train_labels=train_labels, 
                direction_method=direction_method,
                batch_size=batch_size,
            )

        activations = {}

        layer_num = self.model.config.num_hidden_layers

        for layer in layers_id:
            ind = layer - layer_num
            activation = torch.tensor(coeff * rep_reader.directions[ind] * rep_reader.direction_signs[ind], dtype=self.model.dtype).to(self.model.device)
            activations[layer] = activation

        return activations
    
    def get_response(self, prompt, activations=None, max_length = 2000 ):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with self.lock:
            if activations:
                layers_id = list(activations.keys())
                self.wrapped_model.wrap_block(layers_id, block_name="decoder_block")
                self.wrapped_model.set_controller(layers_id, activations, masks=1)

            self.wrapped_model.eval()
            with torch.no_grad():
                outputs = self.wrapped_model.generate(**inputs, do_sample = True, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
                
            self.wrapped_model.unwrap()
        
        texts = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        return texts

    def train(self, data_loader):

        
        pass

    def evaluate(self, data_loader):
        pass

    def save(self, output_dir):
        
        if hasattr(self.model, "module"):
            self.model.module.config.save_pretrained(output_dir)
            self.model.module.save_pretrained(output_dir)
        else:
            self.model.config.save_pretrained(output_dir)
            self.model.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(f'{output_dir}/{self.model_name}')

    def __str__(self):
        return f'LLM(model={self.model_name})'

    def __repr__(self):
        return str(self)
    
class LLaMA2(LLM):
        
    def __init__(self, model_name, device = "cuda", bf16 = False):
        
        super().__init__(model_name, device, bf16)

    def format_prompt(self, msgs:list, assist_prefix:str=""):
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

    def extract_response(self, output:str, assist_prefix:str=""):
        """
        llama2-chat output format:
            "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST] {model_reply}</s><s>[INST] {user_message_2} [/INST] {model_reply}</s>"
        Note:
            </s> will be ignored by the model
        """
        #extract last model replay
        return output.split(f"[/INST] {assist_prefix}")[-1].split("[/INST]")[-1].strip()

class LLaMA3(LLM):
    
    def __init__(self, model_name, device = "cuda", bf16 = False):
        
        super().__init__(model_name, device, bf16)

    def format_prompt(self, msgs, assist_prefix:str=""):
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
    
    def extract_response(self, output, assist_prefix:str=""):
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