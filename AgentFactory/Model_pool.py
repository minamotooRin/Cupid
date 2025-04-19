import importlib

def get_class(path):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class ModelPool:
    def __init__(self):

        self.type2path = {
            "LLaMA3": "AgentFactory.Models.LLaMA3.LLaMA3",
            "DeepSeek": "AgentFactory.Models.DeepSeek.DeepSeek",
            "Qwen": "AgentFactory.Models.Qwen.Qwen",
            "Gemini": "AgentFactory.Models.Gemini.Gemini",
            "LLaVA": "AgentFactory.Models.LLaVA.LLaVA",
            "GPT": "AgentFactory.Models.GPT.GPT",
        }

        self.models = {}
    
    def __getitem__(self, model_name):
        return self.models[model_name]

    def _get_model_class(self, model_type):
        path = self.type2path[model_type]
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def add_local_model(self, model_type, model_name, device, bf16 = True):
        if model_name in self.models:
            return
        model_class = self._get_model_class(model_type)
        self.models[model_name] = model_class(model_name, device, bf16)

    def add_remote_model(self, model_type, model_name, api_key):
        if model_name in self.models:
            return
        model_class = self._get_model_class(model_type)
        self.models[model_name] = model_class(model_name, api_key)
    
    def remove_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]