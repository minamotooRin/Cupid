from AgentFactory.Agent import Agent
from AgentFactory.Model_pool import ModelPool
from AgentFactory.utils import seed_everything

class AgentFactory:
    def __init__(self, seed = 42):
        self.model_pool = ModelPool()
        seed_everything(seed)
    
    def create_agent(self, config):

        if "api_key" in config:
            self.model_pool.add_remote_model(config["template_type"], config["model_name"], config["api_key"])
        else:
            self.model_pool.add_local_model(config["template_type"], config["model_name"], config["device"], config["bf16"])

        agent = Agent(self.model_pool[config["model_name"]])

        return agent