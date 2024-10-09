import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Model name on hugginface")
parser.add_argument("--gpuid", type=str, default="0", help="GPUid to be deployed")
parser.add_argument("--port", type=int, default=5008, help="the port")
parser.add_argument("--gate_path", type=str,help="lora fusion gate ckpt path")
parser.add_argument("--language_lora_path", type=str, default="", help="path to language lora")
parser.add_argument("--task_lora_path", type=str, default="", help="path to task lora")
parser.add_argument("--temperature", type=float, default=1.0, help="temperature of fusion gate")
args = parser.parse_args()

import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftConfig, PeftModel

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("model load finished")

app = Flask(__name__)


def Generate(prompts,model):
    inputs = tokenizer(
        prompts,
        max_length=2048,
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    # decoding
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"], 
        max_new_tokens=512,
        do_sample=True,
        temperature=0.0001,
        top_p=1, 
    )
        
    outputs = tokenizer.batch_decode(
        outputs.to("cpu"), skip_special_tokens=True
    )
    
    for i in range(len(outputs)):
        if outputs[i].startswith(prompts[i]):
            outputs[i] = outputs[i][len(prompts[i]):]
    
    return outputs

def load_gate_model():
    def find_numbers_in_string(text):
        numbers = re.findall(r'\d+', text)
        return int(numbers[0])

    def load_fusion_gate(model=None,gate_path=args.gate_path):
        gate_weight_dict = torch.load(gate_path)
        gate_dict = {k:v for k,v in gate_weight_dict.items() if "lora_fusion_gate" in k}
        weight_bias_dict = {k:v for k,v in gate_weight_dict.items() if "weight_bias" in k}
        
        gate_weight_list , weight_bias_list = [] ,[]
        for i in range(len(gate_dict.keys())):
            layer_gate_key = f"encoder.layers.{i}.lora_fusion_gate.weight"
            weight_bias_key = f"encoder.layers.{i}.weight_bias"
            
            weight_bias_list.append(weight_bias_dict[weight_bias_key])
            gate_weight_list.append(gate_dict[layer_gate_key])
        
        for n,p in model.named_parameters():
            if "lora_fusion_gate" in n or "weight_bias" in n:
                layer = find_numbers_in_string(n)
                if "weight_bias" in n:
                    p.data = weight_bias_list[layer]
                    
                else:
                    p.data = gate_weight_list[layer]

    
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.model_name
            ).to(device)

    # inference temperature = 1.1 for zh_math gate and ru_math gate, 1.0 for others.
    temperature = 1.1 if "zh_math" in args.gate_path or "ru_math" in args.gate_path else 1.0
    for i in range(32):
        ## set temperature
        pretrained_model.model.layers[i].temperature = temperature
        
    lora_model_name_or_path = args.language_lora_path

    #### initialize peftmodel and load the first lora.
    loraflow_model = PeftModel.from_pretrained(pretrained_model, model_id = lora_model_name_or_path, adapter_name = "zh")
    loraflow_model = loraflow_model.to(device)

    #### load other loras.
    loraflow_model.load_adapter(model_id = args.task_lora_path, adapter_name = "math") 

    loraflow_model.base_model.set_adapter(["zh","math"])
    loraflow_model.base_model.model.model.to(device)
    
    load_fusion_gate(loraflow_model)
    
    loraflow_model.to(device).to(pretrained_model.dtype)
    return loraflow_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_gate_model()

@app.route("/url-infer", methods=["POST"])
def main():
    datas = request.get_json()
    prompts = datas["instances"]
    outputs = Generate(prompts=prompts,model=model)
        
    res = []
    for output in outputs:
        generated_text = output
        res.append(generated_text)
    return jsonify(res)


if __name__ == "__main__":
    app.run(port=args.port, debug=False)