import torch
import random
import numpy
import json

def save_to_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def load_jsonl(path):
    with open(path, "r") as f:
        data = f.readlines()
    return [json.loads(d) for d in data]

def seed_everything(seed):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(0)
    random.seed(0)
