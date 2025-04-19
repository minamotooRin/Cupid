from AgentFactory.Models.LLM import LLM
from AgentFactory.utils import save_to_jsonl

class Agent:
    def __init__(self, llm:LLM, instruction:str = None):
        
        self.llm = llm

        self.instruction = instruction

        self.images = []
        self.msg_history = []

        self.reset_msg_history()

    @property
    def instruction(self):
        return self._instruction 
    
    @instruction.setter
    def instruction(self, instruction):
        self._instruction = instruction
        self.reset_msg_history()
    
    @property
    def msgs(self):
        return self.msg_history 
    
    @property
    def controller_texts(self):
        return self.controller_texts 

    def set_llm(self, llm):
        self.llm = llm
    
    def reset_msg_history(self):
        self.msg_history = [{
                "role": self.llm.roleNames["system"],
                "content": [{"type": "text", "text": self.instruction}]
            }] if self.instruction else []

        self.images = []

    def add_demonstration(self, msg, images, response):
        """
        ### Adding ONE demonstration

        msg = [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        images = [image1, image2, ...]
        response = [
                {"type": "text", "text": "There is a red stop sign in the image."},
            ],
        """
        img_cnt = 0
        for content in msg:
            if content["type"] == "image":
                img_cnt += 1
        assert img_cnt == len(images)
        
        self.msg_history.append({
                "role": self.llm.roleNames["user"],
                "content": msg
            })
        self.images += images

        self.msg_history.append({
                "role": self.llm.roleNames["assistant"],
                "content": response
            })
        
    def add_demonstrations(self, demons):
        """
        ### Adding multiple demonstrations
        demons = [
                {
                    "msg": [
                            {"type": "image"},
                            {"type": "text", "text": "What is shown in this image?"},
                        ],
                    "images": [image1, image2, ...],
                    "response": [
                            {"type": "text", "text": "There is a red stop sign in the image."},
                        ],
                },
                ...
            ]
        """

        success_cnt = 0
        for demon in demons:
            try:
                msg = demon["msg"]
                images = demon["images"] if "images" in demon else []
                response = demon["response"]
                self.add_demonstration(msg, images, response)
                success_cnt += 1
            except Exception as e:
                continue
        
        return success_cnt

    def get_response(self, msg, images = [], reply_prefix  = "", max_length = 2000, **kwargs):
        """
        msg = [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        images = [image1, image2, ...]
        """

        img_cnt = 0
        for content in msg:
            if content["type"] == "image":
                img_cnt += 1
        assert img_cnt == len(images), f"Number of images in message ({img_cnt}) does not match the number of images provided ({len(images)})."
        
        self.msg_history.append({
                "role": self.llm.roleNames["user"],
                "content": msg
            })
        self.images += images
        
        response, text = self.llm.get_response(self.msg_history, self.images, max_length = max_length, reply_prefix = reply_prefix, **kwargs)

        self.msg_history.append({
                "role": self.llm.roleNames["assistant"],
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            })

        return response, text
    
    def save_history(self, path):
        save_to_jsonl(path, self.msgs)
    
    def __str__(self):
        return f'Agent(instruction={self.instruction}, llm={self.llm})'

    def __repr__(self):
        return str(self)