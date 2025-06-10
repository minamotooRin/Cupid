import base64
import anthropic
import mimetypes
import os

from AgentFactory.Models.LLM import MLLM_remote

def get_image_base64(path):
    """
    读取本地图片文件，返回 (base64_string, media_type)
    例如: ("...base64...", "image/jpeg")
    """
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            mime_type = 'image/jpeg'
        elif ext == '.png':
            mime_type = 'image/png'
        elif ext == '.gif':
            mime_type = 'image/gif'
        else:
            mime_type = 'application/octet-stream'

    with open(path, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("utf-8")

    return b64_str, mime_type

class Claude(MLLM_remote):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: str = None,
        instr: str = "You are a helpful assistant.",
    ):
        super().__init__(model_name, api_key)

        self.roleNames = {
            "system": "system",
            "user": "user",
            "assistant": "assistant"
        }
        self.client = anthropic.Anthropic(api_key=api_key)
        self.instr = instr

        self.PRICE = {
            "claude-sonnet-4-20250514": {
                "input": 1.5 / 1000000,
                "output": 7.5 / 1000000,
            },
        }
    
    def upload_to_claude(self, path):
        with open(path, "rb") as f:
            file_upload = self.client.beta.files.upload(file=("image.jpg", f, "image/jpeg"))
        return file_upload

    def set_instruction(self, instr: str):
        if instr != self.instr:
            self.instr = instr

    def total_cost(self) -> float:
        p = self.PRICE.get(self.model_name, {})
        return self.input_tokens * p.get("input", 0.0) + self.output_tokens * p.get("output", 0.0)

    def get_response(self, msgs, images, max_length: int = 0, reply_prefix: str = ""):
        """
        [
            {"role":"system"/"user"/"assistant", "content":[{"type":"text"/"image", "text":...,"data":...}]},
            ...
        ]
        images: 本地图片路径列表
        """

        if msgs and msgs[0]["role"] == self.roleNames["system"]:
            sys_instr = "".join(p["text"] for p in msgs[0]["content"] if p["type"] == "text")
            self.set_instruction(sys_instr)

        chat_history = []
        img_idx = 0
        start = 1 if msgs and msgs[0]["role"] == self.roleNames["system"] else 0

        for m in msgs[start:]:
            content = []
            for c in m["content"]:
                if c["type"] == "text":
                    content.append({"type": "text", "text": c["text"]})
                elif c["type"] == "image":
                    # file = self.upload_to_claude(images[img_idx])
                    # content.append({"type": "image", "source":{ "type": "file","file_id": file.id}})
                    b64_str, mime_type = get_image_base64(images[img_idx])
                    content.append({"type": "image", "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_str,
                    }})
                    img_idx += 1
            chat_history.append({"role": m["role"], "content": content})

        # 最终调用 Anthropic 完成
        message = self.client.messages.create(
            model=self.model_name,
            messages=chat_history,
            system=self.instr,
            max_tokens=max_length or None,
        )

        # 更新 token 计数
        self.input_tokens += message.usage.input_tokens
        self.output_tokens += message.usage.output_tokens

        # 返回完整响应对象和文本内容
        text = message.content[0].text
        return message, text
