import random

def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    description = "[INST]为这个问题创建一个Python脚本: " + data["text"] + "[/INST]"
    tests = "\n".join(data["test_list"])

    return {
        "input": f'"""{description}\n{tests}"""',
        "output": data["code"],
        "processed_output": data["code"],
    }
