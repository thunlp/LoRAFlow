from transformers import AutoModelForCausalLM
from peft import LoraModel, LoraConfig
import pdb
import torch
import torch.nn as nn
import sys
sys.path.append("/home/wanghanqing/projects/peft-group_lora/src/peft/tuners/group_lora")

from model import group_LoraModel

model = AutoModelForCausalLM.from_pretrained("/home/wanghanqing/projects/models/Llama-2-7b-chat-hf")

lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

group_LoraModel(model, lora_config, "zh")
# lora_model_1 = group_LoraModel(model, lora_config, "zh")
# lora_model_2 = group_LoraModel(model, lora_config, "code")


# lora_model_2.save_pretrained("/home/wanghanqing/projects/models/Llama-2-7b-chat-hf-group_lora") #model.layers

# def load_group_weights_in_peft(model):
#     from collections import OrderedDict
#     model.load_state_dict(torch.load("/home/wanghanqing/projects/exp/LoRAs/hf_LoRAs/zh_code_gate/adapter_model.bin"),strict=False)
#     zh_weights = torch.load("/home/wanghanqing/projects/exp/LoRAs/hf_LoRAs/zh/adapter_model.bin")
#     code_weights = torch.load("/home/wanghanqing/projects/exp/LoRAs/hf_LoRAs/code/adapter_model.bin")
#     zh = OrderedDict()
#     for key in zh_weights.keys():
#         zh[key.rsplit('.', 1)[0] + ".zh.weight"] = zh_weights[key]
#     code = OrderedDict()
#     for key in code_weights.keys():
#         code[key.rsplit('.', 1)[0] + ".code.weight"] = code_weights[key]
#     model.load_state_dict(zh,strict=False)
#     model.load_state_dict(code,strict=False)

# load_group_weights_in_peft(model)
# 假设 model 是你要检查的模型

# 定义需要检查的组件
# elder_gate = torch.load("/home/wanghanqing/projects/exp/LoRAs/group/group_attn/41000/pytorch_model.bin")
# elder_gate['encoder.layers.31.self_att.self_attention.project_v.lora.attention_gate_V.lora_B']
# layers = range(32)  # layers从0到31
# projs = ['q_proj', 'v_proj']
# gates = ['attention_gate_Q', 'attention_gate_K', 'attention_gate_V']

# # 遍历并检查
# for layer_idx in layers:
#     for proj in projs:
#         for gate in gates:
#             # 构建属性名
#             attribute_name = f"model.model.base_model.layers[{layer_idx}].self_attn.{proj}.{gate}.lora_B.sum()"
#             try:
#                 # 使用 eval 函数动态获取属性值
#                 sum_value = eval(attribute_name)
#                 # 检查sum是否为0
#                 if sum_value == 0:
#                     print(attribute_name)
#             except AttributeError:
#                 # 如果属性不存在，则忽略错误
#                 pass

# import pdb
# pdb.set_trace()


# # model.save_pretrained("/home/wanghanqing/projects/models/Llama-2-7b-chat-hf-group_lora_base")

# model_path = "/home/wanghanqing/projects/models/Llama-2-7b-chat-hf-group_lora_base"

# model_path = "/home/wanghanqing/projects/models/Llama-2-7b-chat-hf-group_lora"
# part1 = torch.load(model_path + '/pytorch_model-00001-of-00003.bin')
# part2 = torch.load(model_path + '/pytorch_model-00002-of-00003.bin')
# part3 = torch.load(model_path + '/pytorch_model-00003-of-00003.bin')

# combined_state_dict = {**part1, **part2, **part3}

# for param_name in combined_state_dict.keys():
#     print(param_name)


# pdb.set_trace()

# model_path = "/home/wanghanqing/projects/models/Llama-2-7b-chat-hf-group_lora"

# part1 = torch.load(model_path + '/pytorch_model-00001-of-00003.bin')
# part2 = torch.load(model_path + '/pytorch_model-00002-of-00003.bin')
# part3 = torch.load(model_path + '/pytorch_model-00003-of-00003.bin')

# combined_state_dict = {**part1, **part2, **part3}

# for param_name in combined_state_dict.keys():
#     print(param_name)