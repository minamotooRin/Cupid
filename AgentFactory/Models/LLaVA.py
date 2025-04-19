import torch
import re
import io
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration

from AgentFactory.Models.LLM import MLLM_local

def load_Image(image_path):

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

class LLaVA(MLLM_local):
    
    def __init__(self, model_name, device = "cuda", bf16 = True):

        super().__init__(model_name, device, bf16)
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16 if bf16 else torch.float32, 
            low_cpu_mem_usage=True, 
        ).to(device)

    def get_response(self, msgs, images, max_length = 2000,):
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

        img_cnt = 0
        for msg in msgs:
            for content in msg["content"]:
                if content["type"] == "image":
                    img_cnt += 1

        assert img_cnt == len(images)

        images_raw = [load_Image(image) for image in images]
        
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.processor(images=images_raw, text=prompt, padding=True, return_tensors="pt").to(self.device)

        with self.lock:
            self.model.eval()
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=max_length, do_sample=False, pad_token_id=self.processor.tokenizer.eos_token_id)
                response = self.processor.decode(output[0], skip_special_tokens=False)
                text = self.extract_response(response)

        return response, text

    def extract_response(self, text):
        """
        llava-hf/llava-interleave-qwen-7b-hf output format:
            "<|im_start|>user {image_tokens}\nWhat are these?|im_end|><|im_start|>assistant"
        Note:
        
        """
        #extract last model replay between "<|start_header_id|>assistant<|end_header_id|>" and "<|eot_id|>"
        pattern = r"<\|im_start\|>assistant(.+?)<\|im_end\|>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None