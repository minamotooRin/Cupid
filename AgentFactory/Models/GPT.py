import base64

from openai import OpenAI

def load_image(image_path):
    # load image to mime_type="image/jpeg" and base64 encode

    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    return b64_image

from AgentFactory.Models.LLM import MLLM_remote

class GPT(MLLM_remote):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name, api_key, instr = "You are a helpful assistant."):
        super().__init__(model_name, api_key)

        self.roleNames = {
            "system": "assistant",
            "user": "user",
            "assistant": "assistant"
        }

        self.instr = instr

        self.PRICE = {
            "gpt-4o":
            {
                "input": 5/1000000,
                "output": 20/1000000,
            },
            "openai-o3":
            {
                "input": 10/1000000,
                "output": 40/1000000,
            },
            "gpt-4.1":
            {
                "input": 2/1000000,
                "output": 8/1000000,
            },
        }

        self.client = OpenAI(
            api_key=api_key,
        )

    def set_instruction(self, instr):
        
        self.instr = instr
    
    def total_cost(self):
        return self.input_tokens * self.PRICE[self.model_name]["input"] + self.output_tokens * self.PRICE[self.model_name]["output"]
    
    def get_response(self, msgs, images, max_length = 0, reply_prefix = ""):

        """
        msgs: [
            {
                'role': 'assistant', 
                'content': [
                    {
                    'type': 'text', 
                    'text': 'You are a helpful assistant.'
                    }
                ]
            }, 
            {
                'role': 'user', 
                'content': [
                    {
                        'type': 'text', 
                        'text': 'Are the two images same?'
                    }, 
                    {'type': 'image'}, 
                    {'type': 'image'}
                ]
            }
        ]
        """

        instr = "".join([it["text"] for it in msgs[0]["content"]]) if msgs[0]["role"] == self.roleNames["system"] else ""
        self.set_instruction(instr)

        img_cnt = 0

        history = []
        for msg in msgs:

            demon = []

            for content in msg["content"]:
                if content["type"] == "image":
                    demon.append({
                        "type": "input_image" if msg["role"] == self.roleNames["user"] else "output_image",
                        "image_url": f"data:image/jpeg;base64,{load_image(images[img_cnt])}"
                    })
                    img_cnt += 1
                elif content["type"] == "text":
                    demon.append({
                        "type": "input_text" if msg["role"] == self.roleNames["user"] else "output_text",
                        "text": content["text"]
                    })

            history.append({"role": self.roleNames[msg["role"]], "content": demon})
        
        response = self.client.responses.create(
            model = self.model_name,
            input = history
        )
        self.input_tokens = response.usage.input_tokens
        self.output_tokens = response.usage.output_tokens

        return response, response.output_text
            
    

