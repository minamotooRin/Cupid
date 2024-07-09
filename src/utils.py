import re
import torch
import random
import numpy

def remove_extra_spaces(text):
    # 使用正则表达式替换多个连续的空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def seed_everything(seed):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(0)
    random.seed(0)
