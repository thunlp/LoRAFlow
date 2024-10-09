from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch
import re
###peft version=0.7.1 更低的版本没有测试过

def find_numbers_in_string(text):
    # 使用正则表达式找到所有数字
    numbers = re.findall(r'\d+', text)
    return int(numbers[0])

def load_fusion_gate(model=None,gate_path="./saved_model/gate.pt"):
    gate_weight_dict = torch.load(gate_path)
    # print(f"{gate_weight_dict}")
    gate_weight_dict = list(gate_weight_dict.values())
    
    for n,p in model.named_parameters():
        if "lora_fusion_gate" in n:
            layer = find_numbers_in_string(n)
            p.data = gate_weight_dict[layer]


if __name__=="__main__":
    #### 载入pretrained_model
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        "/home/wanghanqing/projects/models/Llama-2-7b-hf"
            )
    tokenizer = AutoTokenizer.from_pretrained("/home/wanghanqing/projects/models/Llama-2-7b-hf")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    lora_model_name_or_path = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/zh/lora"

    #### 初始化PeftModel, 并且load第一个adapter
    lora_model = PeftModel.from_pretrained(pretrained_model, model_id = lora_model_name_or_path, adapter_name = "zh")
    lora_model = lora_model.to("cuda")

    # (Pdb) type(lora_model)
    # <class 'peft.peft_model.PeftModelForCausalLM'>
    # (Pdb) type(lora_model.base_model)
    # <class 'peft.tuners.lora.model.LoraModel'>
    # (Pdb) type(lora_model.base_model.model)
    # <class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>
    # (Pdb) type(lora_model.base_model.model.model)
    # <class 'transformers.models.llama.modeling_llama.LlamaModel'>

    #### 读取另外两个adapter
    lora_model.load_adapter(model_id = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/math_epoch3/lora",adapter_name = "math")

    prompts = ["[INST] Hello, who are you? [/INST]"]
    lora_model.base_model.set_adapter(["zh","math"])
    # import pdb
    # pdb.set_trace()
    lora_model.base_model.model.model.to("cuda")

    #### 激活新建的adapter，使得forward时只加上它的lora结果
    #### 激活多个adapter会使得forward时加入多个lora的lora results
    lora_model.base_model.set_adapter(["en-zh-math"])

    #### 把lora的参数merge到模型本体上
    #### 不指定adapter_names的话默认只merge active adapters, 这里指不指定都一样
    merged_model = lora_model.base_model.merge_and_unload(adapter_names = ["en-zh-math"])

    load_fusion_gate(model=merged_model)
    
    merged_model.save_pretrained("/data/public/saved_model/lora_fusion_model/")

    model = AutoModelForCausalLM.from_pretrained(
        "/data/public/saved_model/lora_fusion_model/")
    
    # print(f"{dict(model.named_parameters())['model.layers.0.lora_fusion_gate.weight']}")
    


