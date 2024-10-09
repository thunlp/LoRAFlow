import re

def find_numbers_in_string(text):
    # 使用正则表达式找到所有数字
    numbers = re.findall(r'\d+', text)
    return int(numbers[0])

# 测试字符串
text = "model.layers.0.lora_fusion_gate.weight"
numbers = find_numbers_in_string(text)

# 打印找到的数字
print(numbers)  # 输出可能类似于 ["0"]
