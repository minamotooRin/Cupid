# poetry run python3 -u src/main.py --config_path configs/agent_conf1.json

import fire
import json
import logging

from utils import *
from Agent import Agent
from LLM import ModelPool
from dataset_adapter import dataset_func
# from eval import eval

def main(
        config_path: str = "configs/agent_conf1.json",
):
    seed_everything(42)

    config = json.load(open(config_path, "r"))
    config_1 = config["model_1"]
    config_2 = config["model_2"]
    config_dataset = config["dataset"]

    model_pool = ModelPool()
    model_pool.add_model(config_1["template_type"], config_1["model_name"], config_1["device"])
    model_pool.add_model(config_2["template_type"], config_2["model_name"], config_2["device"])

    agent_1 = Agent(model_pool[config_1["model_name"]])
    agent_2 = Agent(model_pool[config_2["model_name"]])

    questions, answers = dataset_func[config_dataset["dataset_name"]]()

    for question, answer in zip(questions, answers):

        epoch = 1
        scores = []
        terminate_flag = False

        logging.info(f"Question: {question}")
        logging.info(f"Ground Truth: {answer}")
        logging.info(f" ======== Epoch: {epoch} ========")

        agent_1.instruction = config_1["system_msg"]
        response_1 = agent_1.get_response(question, reply_prefix=config_1["answer_prefix"])
        scores.append(1) # eval
        logging.info(f"### Model 1 Output: {response_1}")
        
        agent_2.instruction = config_2["system_msg"].format(question=question)
        response_2 = agent_2.get_response(response_1, reply_prefix=config_2["answer_prefix"])
        logging.info(f"### Model 2 Output: {response_2}")
        
        if remove_extra_spaces(response_2) == config_2["end_msg"] or remove_extra_spaces(response_2) == config_2["answer_prefix"] + config_2["end_msg"]:
            terminate_flag = True

        while terminate_flag == False:

            epoch += 1
            logging.info(f"======== Epoch: {epoch} ========")
            
            response_1 = agent_1.get_response(response_2, reply_prefix=config_1["answer_prefix"])
            scores.append(1) # eval
            logging.info(f"### Model 1 Output: {response_1}")
            
            response_2 = agent_2.get_response(response_1, reply_prefix=config_2["answer_prefix"])
            logging.info(f"### Model 2 Output: {response_2}")
            
            if remove_extra_spaces(response_2) == config_2["end_msg"] or remove_extra_spaces(response_2) == config_2["answer_prefix"] + config_2["end_msg"]:
                terminate_flag = True
                
            if epoch == config["max_iter"]:
                terminate_flag = True

        logging.info(f"======== END IN: {epoch} ========")
        
        break # for debugging, just test one case


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)