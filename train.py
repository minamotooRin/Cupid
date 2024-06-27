#nohup poetry run accelerate launch train.py --config_path configs/train_conf3.json > log/train_conf3.log 2>&1 &

import fire
import torch
import random
import json
import logging


from tqdm import tqdm
from pathlib import Path

from dataset_adapter import dataset2tensor_func
from format import model2format_func

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from accelerate import Accelerator

def main(
        config_path: str = "configs/train_conf1.json",
):
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. Please check your setup.")

    config = json.load(open(config_path, "r"))
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    dataset_path = config["dataset"]
    output_dir = config["save_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )

    model = AutoModelForCausalLM.from_pretrained(config["model"], trust_remote_code=True)
    model = get_peft_model(model, peft_config)
    model.bfloat16() # may cause error in training, or results in nan loss, refer to https://zhuanlan.zhihu.com/p/671165275
    max_len = model.config.max_position_embeddings

    format_func  = model2format_func[config["model"]]["encode"]
    extract_func = model2format_func[config["model"]]["decode"]
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logging.info("Model is loaded.")

    train_datasets = []
    test_datasets = []
    total_train_tokens = 0
    total_test_tokens = 0
    
    instructions = config["instructions"]

    for dataset_path in config["dataset"]:
        # NOTE: max_len//4 is used to avoid OOM, 和解了
        tr_d, train_tokens, te_d, test_tokens = dataset2tensor_func[dataset_path](instructions, tokenizer, max_len//4, format_func)
        train_datasets.append(tr_d)
        test_datasets.append(te_d)
        total_train_tokens += train_tokens - len(te_d) # remove the last token, which is eos
        total_test_tokens += test_tokens - len(te_d) # remove the last token, which is eos
        
    # Combine multiple data loaders into one
    train_loader = DataLoader(torch.utils.data.ConcatDataset(train_datasets), shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(torch.utils.data.ConcatDataset(test_datasets), shuffle=False, batch_size=batch_size)
    logging.info("Dataset are loaded.")

    loss_fct = CrossEntropyLoss(reduction="sum")
    num_training_steps = len(train_loader) * num_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
    accelerator = Accelerator()
    model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare( model, optimizer, train_loader, test_loader, lr_scheduler )

    def masked_loss(input_ids, attention_mask, logits, start_ids, end_ids):
        labels = input_ids[:, 1:].contiguous()

        batch_size, seq_len, vocab_size = logits.size()

        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(accelerator.device) >= start_ids.unsqueeze(1)

        mask = attention_mask.bool() & mask # MUST BE BOOL, NOT 0 and 1 !!!

        # 最后一个True代表最后一个有效token（即eos），并不参与loss计算，应该去掉。由于padding的存在，去掉的手续等价为将其置为False，然后去掉最后一个False
        mask[torch.arange(batch_size), (end_ids - 1)] = False
        mask = mask[:, :-1]

        shift_logits = logits[:, :-1, :][mask]
        shift_labels = labels[mask]

        loss = loss_fct(shift_logits, shift_labels)
        token_num = mask.sum().item()
        return loss, token_num

    def evaluate(test_loader):
        
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:

                input_ids = batch[0]
                attention_mask = batch[1]

                start_ids = batch[2]
                end_ids = attention_mask.sum(dim=1)
                max_ids = torch.max(end_ids)

                input_ids = input_ids[:, :max_ids]
                attention_mask = attention_mask[:, :max_ids]

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Manually compute loss, not so fast but works
                # labels = input_ids.clone()
                # loss = 0
                # for i in range(len(batch[0])):
                #     shift_logits = logits[i, start_ids[i]    : end_ids[i] - 1, :]
                #     shift_labels = labels[i, start_ids[i] + 1: end_ids[i]    ]
                #     loss += loss_fct(shift_logits, shift_labels)

                loss, token_num = masked_loss(input_ids, attention_mask, logits, start_ids, end_ids)

                if torch.isnan(loss):
                    logging.info("Loss is nan.")
                    continue

                total_test_loss += loss.item()

        return total_test_loss
    
    def train_step(batch):

        input_ids = batch[0]
        attention_mask = batch[1]

        start_ids = batch[2]
        end_ids = attention_mask.sum(dim=1)
        max_ids = torch.max(end_ids)

        input_ids = input_ids[:, :max_ids]
        attention_mask = attention_mask[:, :max_ids]

        try:
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Manually compute loss, not so fast but works
        # labels = input_ids.clone()
        # loss = 0
        # token_num = 0
        # for i in range(len(batch[0])):
        #     shift_logits = logits[i, start_ids[i]    : end_ids[i] - 1, :]
        #     shift_labels = labels[i, start_ids[i] + 1: end_ids[i]    ]
        #     loss += loss_fct(shift_logits, shift_labels)
        #     token_num += end_ids[i] - 1 - start_ids[i]
        # loss /= token_num
        
            loss, token_num = masked_loss(input_ids, attention_mask, logits, start_ids, end_ids)

            reducted_loss = loss / token_num
            if torch.isnan(reducted_loss):
                logging.info("Loss is nan.")
                return loss.item()

            accelerator.backward(reducted_loss)
    
        except Exception as e:
            logging.error(f"Error occurs in model: {e}")
            logging.error(f"Input ids: {input_ids.shape}")
            logging.error(f"Attention mask: {attention_mask.shape}")
            logging.error(f"Start ids: {start_ids}")
            logging.error(f"End ids: {end_ids}")
            exit(0)

        return loss.item()

    logging.info("Start training ...")

    model.eval()
    with torch.no_grad():
        total_test_loss = evaluate(test_loader)
        avg_test_loss = total_test_loss / total_test_tokens
        logging.info(f"Epoch 0/{num_epochs} | Loss: - | Validation Loss: {avg_test_loss:.4f}")

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for batch in train_loader:
            
            optimizer.zero_grad()
            loss = train_step(batch)
            total_loss += loss
            optimizer.step()
            lr_scheduler.step() # update learning rate each epoch?

            progress_bar.update(1)

        model.eval()
        with torch.no_grad():
            total_test_loss = evaluate(test_loader)
            avg_test_loss = total_test_loss / total_test_tokens

        avg_loss = total_loss / total_train_tokens
        logging.info(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Validation Loss: {avg_test_loss:.4f}")

        model_config = model.module.config if hasattr(model, "module") else model.config
        model_config.save_pretrained(f'{output_dir}/checkpoint-{epoch}')
        model.module.save_pretrained(f'{output_dir}/checkpoint-{epoch}')

    tokenizer.save_pretrained(f'{output_dir}/checkpoint-{epoch}')
    logging.info("Training is finished.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)