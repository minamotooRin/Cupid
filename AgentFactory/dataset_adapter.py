import random
import torch
from tqdm import tqdm

from datasets import load_from_disk
from torch.utils.data import TensorDataset

def construct_prompt(instrs, input_texts, output_texts, tokenizer, max_len, format_func):
    assert tokenizer.padding_side == "left"
    
    prompts = []
    outputs_len = []
    token_num = 0
    for it, ot in tqdm(zip(input_texts, output_texts)):
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

        output_len = len(prompt['input_ids']) - len(prompt_wo_assist['input_ids']) + 1 # prompt['input_ids'][:-output_len]
        
        prompts.append(prompt['input_ids'])
        outputs_len.append(output_len)

        token_num += len(prompt['input_ids']) - len(prompt_wo_assist['input_ids'])

    outputs_len = torch.tensor(outputs_len)
    
    # ============================
    # 优化: TensorDataset需要形状保持一致，导致padding至最长文本。可使用自定义数据集而非TensorDataset，在读取batch后进行padding
    # repley: 在训练时裁剪多余的padding可解决这个问题, e,g:
    # bs, seq_len = input_ids.shape
    # act_len = attention_mask.sum(dim=1)
    # max_ids = act_len.max().item()

    # input_ids = input_ids[:, -max_ids:]
    # attention_mask = attention_mask[:, -max_ids:]
    # repley: 内存容易爆
    # ============================
    
    encoded_inputs = tokenizer.pad({"input_ids": prompts}, return_tensors="pt", padding="max_length", max_length=max_len)
    
    my_dataset = TensorDataset(
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
        outputs_len
    )

    return my_dataset, token_num


class DatasetAdapter:

    def __init__(self):

        self.dataset = {}

    def load_dataset(self, path):

        if path in self.dataset:
            return

        D = load_from_disk(path)
        self.dataset[path] = D


    def get_tokenized_dataset(self, path, instrs, tokenizer, max_len, format_func):

        if path not in self.dataset:
            self.load_dataset(path)

        train_dataset = self.dataset[path]

        def get_dataset(dataset):
            my_dataset, token_num = construct_prompt(instrs, dataset["input"], dataset["output"], tokenizer, max_len, format_func)
            return my_dataset, token_num
        
        train_dataset, train_token_num = get_dataset(train_dataset)

        return train_dataset, train_token_num
