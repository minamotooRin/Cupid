import re

def remove_extra_spaces(text):
    # 使用正则表达式替换多个连续的空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()