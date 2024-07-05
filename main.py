# poetry run python3 -u main.py --config_path configs/agent_conf1.json

import fire
import torch
import json
import logging
import tqdm
import random
import numpy
import pathlib as pl

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *
from format import model2format_func
from dataset_adapter import dataset_func
from repe_wrapper import get_wrapped_model
# from eval import eval

def get_response(model, tokenizer, encode_func, decode_func, prompt, assist_prefix = ""):
    formatted_prompt = encode_func(prompt, assist_prefix)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, do_sample = True, max_length=model.config.max_position_embeddings, pad_token_id=tokenizer.eos_token_id)
    texts = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = decode_func(texts, assist_prefix)
    return response

def seed_everything(seed):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(0)
    random.seed(0)

def main(
        config_path: str = "configs/agent_conf1.json",
):
    seed_everything(42)

    config = json.load(open(config_path, "r"))
    config_1 = config["model_1"]
    config_2 = config["model_2"]
    config_dataset = config["dataset"]
    
    def load_model(config):
        
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForCausalLM.from_pretrained(config["model_name"]).to(config["device"])
        encode_func = model2format_func[config["template_type"]]["encode"]
        decode_func = model2format_func[config["template_type"]]["decode"]
        # if config["wrapper"]:
        #     warp_config = config["wrapper"]
            
        #     # TBD, combine pos_instrs and neg_instrs with dataset load from activation_demons, add into pn_pairs
        #     pos_instrs = warp_config["pos_instrs"]
        #     neg_instrs = warp_config["neg_instrs"]
        #     activation_demons = warp_config["activation_demons"]
        #     pn_pairs = list(zip(pos_instrs, neg_instrs))

        #     layer_id = warp_config["layer_id"]
        #     coeff = warp_config["coeff"]
        #     model = get_wrapped_model(model, tokenizer, pn_pairs, layer_id, coeff)
        
        return model, tokenizer, encode_func, decode_func
    
    model_1, tokenizer_1, encode_func_1, decode_func_1 = load_model(config_1)
    model_2, tokenizer_2, encode_func_2, decode_func_2 = load_model(config_2)

    questions, answers = dataset_func[config_dataset["dataset_name"]]()

    for question, answer in zip(questions, answers):

        epoch = 1

        scores = []
        
        terminate_flag = False

        logging.info(f"Question: {question}")
        logging.info(f"Ground Truth: {answer}")

        logging.info(f" ======== Epoch: {epoch} ========")

        prompts_1 = [
            {
                "system_msg": config_1["system_msg"]
            },
            {
                "user_msg": question
            }
        ]

        response_1 = get_response(model_1, tokenizer_1, encode_func_1, decode_func_1, prompts_1, assist_prefix=config_1["answer_prefix"])
        logging.info(f"Model 1 Output: {response_1}")

        # scores.append(eval(answer, response_1))
        scores.append(1)

        prompts_2 = [
            {
                "system_msg": config_2["system_msg"].format(question = question)
            },
            {
                "user_msg": response_1
            }
        ]
        response_2 = get_response(model_2, tokenizer_2, encode_func_2, decode_func_2, prompts_2, assist_prefix=config_2["answer_prefix"])
        if remove_extra_spaces(response_2) == config_2["end_msg"] or remove_extra_spaces(response_2) == config_2["answer_prefix"] + config_2["end_msg"]:
            terminate_flag = True
        logging.info(f"----------------------------------")
        logging.info(f"Model 2 Output: {response_2}")

        while terminate_flag == False:

            epoch += 1
            logging.info(f"======== Epoch: {epoch} ========")

            prompts_1.append(
                {
                    "assistant_msg": response_1,
                }
            )
            prompts_1.append(
                {
                    "user_msg": response_2,
                }
            )
            response_1 = get_response(model_1, tokenizer_1, encode_func_1, decode_func_1, prompts_1, assist_prefix=config_1["answer_prefix"])
            logging.info(f"Model 1 Output: {response_1}")
            
            # scores.append(eval(answer, response_1))
            scores.append(1)

            prompts_2.append(
                {
                    "assistant_msg": response_2,
                }
            )
            prompts_2.append(
                {
                    "user_msg": response_1,
                }
            )
            response_2 = get_response(model_2, tokenizer_2, encode_func_2, decode_func_2, prompts_2, assist_prefix=config_2["answer_prefix"])
            if remove_extra_spaces(response_2) == config_2["end_msg"] or remove_extra_spaces(response_2) == config_2["answer_prefix"] + config_2["end_msg"]:
                terminate_flag = True

            logging.info(f"----------------------------------")
            logging.info(f"Model 2 Output: {response_2}")

            if epoch == config["max_iter"]:
                terminate_flag = True

        logging.info(f"======== END IN: {epoch} ========")
        
        break # for debugging, just test one case


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)