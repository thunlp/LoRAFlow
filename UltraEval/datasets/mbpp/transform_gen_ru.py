import random

def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    description = "[INST]Создайте Python скрипт для этой задачи: " + data["text"] + "[/INST]"
    tests = "\n".join(data["test_list"])

    return {
        "input": f'"""{description}\n{tests}"""',
        "output": data["code"],
        "processed_output": data["code"],
    }