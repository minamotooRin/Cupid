import random
import torch

from datasets import load_dataset
from torch.utils.data import TensorDataset

def construct_prompt(instrs, input_texts, output_texts, tokenizer, max_len, format_func):
    
    prompts = []
    start_ids = []
    token_num = 0
    for it, ot in zip(input_texts, output_texts):
        instr = random.choice(instrs)
        msgs_wo_assist = [
            {
                "system_msg": instr
            },
            {
                "user_msg": it
            }
        ]
        msgs = [
            {
                "system_msg": instr
            },
            {
                "user_msg": it
            },
            {
                "assistant_msg": ot
            }
        ]
        prompt_wo_assist = format_func(msgs_wo_assist)
        prompt = format_func(msgs)
        
        prompt_wo_assist = tokenizer(prompt_wo_assist, padding=False, truncation=False)
        prompt = tokenizer(prompt, padding=False, truncation=False)

        if len(prompt['input_ids']) > max_len:
            continue

        assist_start_idx = len(prompt_wo_assist['input_ids']) # prompt['input_ids'][assist_start_idx:]
        
        prompts.append(prompt['input_ids'])
        start_ids.append(assist_start_idx)

        token_num += len(prompt['input_ids']) - len(prompt_wo_assist['input_ids'])

    start_ids = torch.tensor(start_ids)
    
    # TBD优化: TensorDataset需要形状保持一致。使用自定义数据集而非TensorDataset，在读取batch后进行padding
    # repley: 在训练时裁剪了多余的padding解决了这个问题
    encoded_inputs = tokenizer.pad({"input_ids": prompts}, return_tensors="pt", padding="max_length", max_length=max_len)
    
    my_dataset = TensorDataset(
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
        start_ids
    )

    return my_dataset, token_num

def QA_dataset_MedQuad():
    d = load_dataset("keivalya/MedQuad-MedicalQnADataset")["train"]
    return d["Question"], d["Answer"]
def get_dataset_MedQuad(instrs, tokenizer, max_len, format_func):

    d = load_dataset("keivalya/MedQuad-MedicalQnADataset")["train"]
    split_dataset = d["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]

    def get_dataset(dataset):

        my_dataset, token_num = construct_prompt(instrs, dataset["Question"], dataset["Answer"], tokenizer, max_len, format_func)

        return my_dataset, token_num
    
    train_dataset, train_token_num = get_dataset(train_dataset)
    valid_dataset, valid_token_num = get_dataset(valid_dataset)

    return train_dataset, train_token_num, valid_dataset, valid_token_num

def QA_dataset_medical_llama3():
    d = load_dataset("Shekswess/medical_llama3_instruct_dataset_short")["train"]
    return d["input"], d["output"]
def get_dataset_medical_llama3(instrs, tokenizer, max_len, format_func):

    d = load_dataset("Shekswess/medical_llama3_instruct_dataset_short")
    split_dataset = d["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]

    def get_dataset(dataset):

        my_dataset, token_num = construct_prompt(instrs, dataset["input"], dataset["output"], tokenizer, max_len, format_func)

        return my_dataset, token_num
    
    train_dataset, train_token_num = get_dataset(train_dataset)
    valid_dataset, valid_token_num = get_dataset(valid_dataset)

    return train_dataset, train_token_num, valid_dataset, valid_token_num


def QA_dataset_PubMedQA():
    pass
def get_dataset_PubMedQA(instrs, tokenizer, max_len, format_func):
    "qiaojin/PubMedQA"
    pass

def QA_dataset_MedQAUSMLE():
    d = load_dataset("GBaker/MedQA-USMLE-4-options")["train"]
    return d["question"], d["options"]
def get_dataset_MedQAUSMLE(instrs, tokenizer, max_len, format_func):

    d = load_dataset("GBaker/MedQA-USMLE-4-options")["train"]
    split_dataset = d.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]

    def get_dataset(dataset):

        input_texts = []
        for q, ops in zip(dataset["question"], dataset["options"]):
            ops_text = '\n'.join([f"{key}: {value}" for key, value in ops.items()])
            input_texts.append(f"{q}\n{ops_text}")
        my_dataset, token_num = construct_prompt(instrs, input_texts, dataset["answer"], tokenizer, max_len, format_func)

        return my_dataset, token_num
    
    train_dataset, train_token_num = get_dataset(train_dataset)
    valid_dataset, valid_token_num = get_dataset(valid_dataset)

    return train_dataset, train_token_num, valid_dataset, valid_token_num

def QA_dataset_MedChinese():
    d = load_dataset("shibing624/medical","finetune")["validation"]
    return d["instruction"], d["output"]
def get_dataset_MedChinese(instrs, tokenizer, max_len, format_func):

    train_dataset = load_dataset("shibing624/medical","finetune")["train"]
    valid_dataset = load_dataset("shibing624/medical","finetune")["validation"]

    def get_dataset(dataset):

        my_dataset, token_num = construct_prompt(instrs, dataset["instruction"], dataset["output"], tokenizer, max_len, format_func)

        return my_dataset, token_num
    
    train_dataset, train_token_num = get_dataset(train_dataset)
    valid_dataset, valid_token_num = get_dataset(valid_dataset)

    return train_dataset, train_token_num, valid_dataset, valid_token_num

def QA_dataset_liveQA():
    d = load_dataset("truehealth/liveqa")["train"]
    return d["message"], d["answer"]
def get_dataset_liveQA(instrs, tokenizer, max_len, format_func):
    
        d = load_dataset("truehealth/liveqa")["train"]
        split_dataset = d.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        valid_dataset = split_dataset["test"]
    
        def get_dataset(dataset):
    
            my_dataset, token_num = construct_prompt(instrs, dataset["message"], dataset["answer"], tokenizer, max_len, format_func)
    
            return my_dataset, token_num
        
        train_dataset, train_token_num = get_dataset(train_dataset)
        valid_dataset, valid_token_num = get_dataset(valid_dataset)
    
        return train_dataset, train_token_num, valid_dataset, valid_token_num

def QA_dataset_medicationqa():
    d = load_dataset("truehealth/medicationqa")["train"]
    return d["question"], d["answer"]
def get_dataset_medicationqa(instrs, tokenizer, max_len, format_func):

    d = load_dataset("truehealth/medicationqa")["train"]
    split_dataset = d.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]

    def get_dataset(dataset):

        my_dataset, token_num = construct_prompt(instrs, dataset["question"], dataset["answer"], tokenizer, max_len, format_func)

        return my_dataset, token_num
    
    train_dataset, train_token_num = get_dataset(train_dataset)
    valid_dataset, valid_token_num = get_dataset(valid_dataset)

    return train_dataset, train_token_num, valid_dataset, valid_token_num

dataset2tensor_func = {
    "keivalya/MedQuad-MedicalQnADataset": get_dataset_MedQuad,
    "Shekswess/medical_llama3_instruct_dataset_short": get_dataset_medical_llama3,
    "qiaojin/PubMedQA": get_dataset_PubMedQA,
    "GBaker/MedQA-USMLE-4-options": get_dataset_MedQAUSMLE,
    "shibing624/medical": get_dataset_MedChinese,
    "truehealth/liveqa": get_dataset_liveQA,
}

dataset_func = {
    "keivalya/MedQuad-MedicalQnADataset": QA_dataset_MedQuad,
    "Shekswess/medical_llama3_instruct_dataset_short": QA_dataset_medical_llama3,
    "qiaojin/PubMedQA": QA_dataset_PubMedQA,
    "GBaker/MedQA-USMLE-4-options": QA_dataset_MedQAUSMLE,
    "shibing624/medical": QA_dataset_MedChinese,
    "truehealth/liveqa": QA_dataset_liveQA,
}