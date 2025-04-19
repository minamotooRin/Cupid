import base64
import google.generativeai as genai

from AgentFactory.Models.LLM import MLLM_remote

def load_image(image_path):
    # load image to mime_type="image/jpeg" and base64 encode

    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    return b64_image

def upload_to_gemini(path, mime_type=None):
    """
    "image/jpeg"
    Uploads the given file to Gemini.
    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

class Gemini(MLLM_remote):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name, api_key, instr = "You are a helpful assistant."):
        super().__init__(model_name, api_key)

        self.roleNames = {
            "system": "model",
            "user": "user",
            "assistant": "model"
        }

        self.instr = instr

        self.PRICE = {
            "gemini-1.5-flash":
            {
                "input": 0.075/1000000,
                "output": 0.3/1000000,
            },
            "gemini-2.0-flash-exp":
            {
                "input": 0.075/1000000,
                "output": 0.3/1000000,
            },
        }

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(self.model_name, system_instruction=self.instr)

    def set_instruction(self, instr):
        if instr == self.instr:
            return
        self.instr = instr
        self.model = genai.GenerativeModel(self.model_name,
            system_instruction=self.instr
        )
    
    def total_cost(self):
        return self.input_tokens * self.PRICE[self.model_name]["input"] + self.output_tokens * self.PRICE[self.model_name]["output"]
    
    def get_response(self, msgs, images, max_length = 0, reply_prefix = ""):
        
        """
        msgs = [
            {
                "role": "model",
                "parts": [
                    {"text": "Identify the similarities between these images."},
                ],
            }
            {
                "role": "user",
                "parts": [
                    {'mime_type':'image/jpeg', "data": "..."},
                    {'mime_type':'image/jpeg', "data": "..."},
                    {"text": "Identify the similarities between these images."},
                ],
            }
        ]
        """
        
        instr = "".join([it["text"] for it in msgs[0]["content"]]) if msgs[0]["role"] == self.roleNames["system"] else ""
        self.set_instruction(instr)

        img_cnt = 0

        history = []
        it = 1 if msgs[0]["role"] == self.roleNames["system"] else 0
        while it + 1 < len(msgs):

            demon_user = []
            for content in msgs[it]["content"]:
                if content["type"] == "image":
                    demon_user.append({'mime_type':'image/jpeg', 'data': load_image(images[img_cnt])})
                    img_cnt += 1
                elif content["type"] == "text":
                    demon_user.append({'text':content["text"]})

            demon_assi = []
            for content in msgs[it + 1]["content"]:
                if content["type"] == "image":
                    demon_assi.append({'mime_type':'image/jpeg', 'data': load_image(images[img_cnt])})
                    img_cnt += 1
                elif content["type"] == "text":
                    demon_assi.append({'text':content["text"]})

            history.append({"role": self.roleNames["user"], "parts": demon_user})
            history.append({"role": self.roleNames["assistant"], "parts": demon_assi})
            
            it += 2

        query = []
        for content in msgs[-1]["content"]:
            if content["type"] == "image":
                query.append({'mime_type':'image/jpeg', 'data': load_image(images[img_cnt])})
                img_cnt += 1
            elif content["type"] == "text":
                query.append({'text':content["text"]})

        if history == []:
            response = self.model.generate_content(query)
        else:
            chat = self.model.start_chat(
                history=history
            )
            response = chat.send_message(query)

        self.input_tokens += response.usage_metadata.prompt_token_count
        self.output_tokens += response.usage_metadata.candidates_token_count

        return response, response.candidates[0].content.parts[0].text
        