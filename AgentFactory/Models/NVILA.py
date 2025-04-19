# TBD

import torch
import re
from transformers import  AutoModel

from AgentFactory.Models.LLM import MLLM_local, MLLM_remote

class NVILA(MLLM_local):
    
    def __init__(self, model_name, device = "cuda", bf16 = True):

        super().__init__(model_name, device, bf16)
        
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype = torch.bfloat16 if bf16 else torch.float32, 
            low_cpu_mem_usage=True, 
        ).to(device)

    def get_response(self, msgs, max_length = 2000,):
        """
        msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "There is a red stop sign in the image."},
                ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What about this image? How many cats do you see?"},
                ],
        },
    ]
        """
        
        inputs = self.processor.apply_chat_template(msgs, add_generation_prompt=True)

        with self.lock:
            self.model.eval()
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=max_length, do_sample=False)
                text = self.processor.decode(output[0], skip_special_tokens=True)

        return text

    def extract_response(self, output):
        """
        nvila output format:
            "<|im_start|>user {image_tokens}\nWhat are these?|im_end|><|im_start|>assistant"
        Note:
        
        """
        #extract last model replay between "<|start_header_id|>assistant<|end_header_id|>" and "<|eot_id|>"
        pattern = r"<\|im_start\|>assistant(.+?)<\|im_end\|>"
        matches = re.findall(pattern, output, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
        

class NVILA(MLLM_remote):

    def __init__(self, model_name = "NVILA-8B", url = "http://localhost:8000"):

        super().__init__(model_name, "fake-key")
        
        self.client =  OpenAI(
            base_url=url,
            api_key="fake-key",
        )

    def get_response(self, msgs, images, max_length = 2000):
        """
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                    ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "There is a red stop sign in the image."},
                    ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What about this image? How many cats do you see?"},
                    ],
            },
        ]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://blog.logomyway.com/wp-content/uploads/2022/01/NVIDIA-logo.jpg",
                            # Or you can pass in a base64 encoded image
                            # "url": "data:image/png;base64,<base64_encoded_image>",
                        },
                    },
                ],
            }
        ]
        """

        idx = 0

        for msg in msgs:
            for content in msg["content"]:
                if content["type"] == "image":
                    content["type"] = "image_url"

                    images[idx].convert("RGBA")
                    b64_img = base64.b64encode(images[idx].tobytes()).decode("utf-8")
                    content["image_url"] = {"url": f"data:image/png;base64,{b64_img}"}
                    idx += 1

        response = self.client.chat.completions.create(
            messages=msgs,
            model=self.model_name,
        )

        return response.choices[0].message.content
    
    def extract_response(self, texts):
        
        return texts

