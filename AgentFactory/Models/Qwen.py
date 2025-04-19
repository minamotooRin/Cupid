import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from AgentFactory.Models.LLM import MLLM_local

class Qwen(MLLM_local):
    "Qwen/Qwen2.5-VL-7B-Instruct"
    
    def __init__(self, model_name, device = "cuda", bf16 = True):

        super().__init__(model_name, device, bf16)

        self.roleNames = {
            "system": "system",
            "user": "user",
            "assistant": "assistant"
        }
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16 if bf16 else torch.float32, 
        ).to(device)

    def get_response(self, msgs, images, max_length = 2000, **kwargs):
        """
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "file:///path/to/image1.jpg"},
                    {"type": "image", "image": "file:///path/to/image2.jpg"},
                    {"type": "text", "text": "Identify the similarities between these images."},
                ],
            }
        ]
        """

        min_pixels = kwargs.get('min_pixels', 256 * 28 * 28)
        max_pixels = kwargs.get('max_pixels', 512 * 28 * 28)

        img_cnt = 0
        for msg in msgs:
            for content in msg["content"]:
                if content["type"] == "image":
                    if img_cnt >= len(images):
                        raise ValueError(f"Too many images in the message. Expected {len(images)} images.")
                    content["image"] = "file://" + images[img_cnt]
                    content["min_pixels"] = min_pixels
                    content["max_pixels"] = max_pixels
                    img_cnt += 1

        assert img_cnt == len(images)
        
        prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = self.processor(images=image_inputs, text=[prompt], padding=True, return_tensors="pt").to(self.device)

        with self.lock:
            self.model.eval()
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                output = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
        return response[0], output[0]
