import sys
import torch
import argparse
import os
import json
tmp_path = "./transfer_tmp.bin"

def filter_only_lora(input_model_path, output_model_path):
    model_state_dict = torch.load(input_model_path)

    # 创建一个新的状态字典，只包含包含特定关键字的参数
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if 'lora' in k}

    # 保存过滤后的状态字典到新文件
    torch.save(filtered_state_dict, output_model_path)

def transfer_malign_lora_2_peft(input_model_path, output_model_path, direction):
    from transformers import LlamaConfig
    import torch, os
    import json
    from collections import OrderedDict
    import sys
    import shutil
    os.makedirs(output_model_path, exist_ok=True)

    ### 7B model
    layernum = 32

    model_hf = OrderedDict()
    if direction == "peft2malign":
        input_name = "adapter_model.bin"
        output_name = "lora.pt"
    elif direction == "malign2peft":
        input_name = "lora.pt"
        output_name = "adapter_model.bin"
    else:
        print(f"Unexpected direction{direction}")
    input_model_path = os.path.join(input_model_path, input_name)
    param = torch.load(input_model_path)
    model_hf.update(param)

    out = OrderedDict()

    for lnum in range(layernum):
        peft_pfx = f"base_model.model.model.layers.{lnum}"
            
        # elif lora_ver == "hf":
        #     peft_pfx = f"model.layers.{lnum}"
        # elif lora_ver == "malign_gate":
        #     peft_pfx = f"encoder.layers.{lnum}"
        # else:
            # raise Exception("lora_ver must be peft or hf")
        delta_pfx = f"encoder.layers.{lnum}"

        if direction == "peft2malign":
            out[f"{delta_pfx}.self_att.self_attention.project_q_lora.lora_A.weight"] = model_hf[f"{peft_pfx}.self_attn.q_proj.lora_A.weight"].contiguous()
            out[f"{delta_pfx}.self_att.self_attention.project_q_lora.lora_B.weight"] = model_hf[f"{peft_pfx}.self_attn.q_proj.lora_B.weight"].contiguous()
            out[f"{delta_pfx}.self_att.self_attention.project_k_lora.lora_A.weight"] = model_hf[f"{peft_pfx}.self_attn.k_proj.lora_A.weight"].contiguous()
            out[f"{delta_pfx}.self_att.self_attention.project_k_lora.lora_B.weight"] = model_hf[f"{peft_pfx}.self_attn.k_proj.lora_B.weight"].contiguous()
            out[f"{delta_pfx}.self_att.self_attention.project_v_lora.lora_A.weight"] = model_hf[f"{peft_pfx}.self_attn.v_proj.lora_A.weight"].contiguous()
            out[f"{delta_pfx}.self_att.self_attention.project_v_lora.lora_B.weight"] = model_hf[f"{peft_pfx}.self_attn.v_proj.lora_B.weight"].contiguous()
            out[f"{delta_pfx}.self_att.self_attention.attention_out_lora.lora_A.weight"] = model_hf[f"{peft_pfx}.self_attn.o_proj.lora_A.weight"].contiguous()
            out[f"{delta_pfx}.self_att.self_attention.attention_out_lora.lora_B.weight"] = model_hf[f"{peft_pfx}.self_attn.o_proj.lora_B.weight"].contiguous()

            out[f"{delta_pfx}.ffn.ffn.w_in.w_0_lora.lora_A.weight"] = model_hf[f"{peft_pfx}.mlp.gate_proj.lora_A.weight"].contiguous()
            out[f"{delta_pfx}.ffn.ffn.w_in.w_0_lora.lora_B.weight"] = model_hf[f"{peft_pfx}.mlp.gate_proj.lora_B.weight"].contiguous()
            out[f"{delta_pfx}.ffn.ffn.w_in.w_1_lora.lora_A.weight"] = model_hf[f"{peft_pfx}.mlp.up_proj.lora_A.weight"].contiguous()
            out[f"{delta_pfx}.ffn.ffn.w_in.w_1_lora.lora_B.weight"] = model_hf[f"{peft_pfx}.mlp.up_proj.lora_B.weight"].contiguous()
            out[f"{delta_pfx}.ffn.ffn.w_out_lora.lora_A.weight"] = model_hf[f"{peft_pfx}.mlp.down_proj.lora_A.weight"].contiguous()
            out[f"{delta_pfx}.ffn.ffn.w_out_lora.lora_B.weight"] = model_hf[f"{peft_pfx}.mlp.down_proj.lora_B.weight"].contiguous()
            
        elif direction == "malign2peft":
            out[f"{peft_pfx}.self_attn.q_proj.lora_A.weight"] = model_hf[f"{delta_pfx}.self_att.self_attention.project_q_lora.lora_A.weight"].contiguous()
            out[f"{peft_pfx}.self_attn.q_proj.lora_B.weight"] = model_hf[f"{delta_pfx}.self_att.self_attention.project_q_lora.lora_B.weight"].contiguous()
            out[f"{peft_pfx}.self_attn.k_proj.lora_A.weight"] = model_hf[f"{delta_pfx}.self_att.self_attention.project_k_lora.lora_A.weight"].contiguous()
            out[f"{peft_pfx}.self_attn.k_proj.lora_B.weight"] = model_hf[f"{delta_pfx}.self_att.self_attention.project_k_lora.lora_B.weight"].contiguous()
            out[f"{peft_pfx}.self_attn.v_proj.lora_A.weight"] = model_hf[f"{delta_pfx}.self_att.self_attention.project_v_lora.lora_A.weight"].contiguous()
            out[f"{peft_pfx}.self_attn.v_proj.lora_B.weight"] = model_hf[f"{delta_pfx}.self_att.self_attention.project_v_lora.lora_B.weight"].contiguous()
            out[f"{peft_pfx}.self_attn.o_proj.lora_A.weight"] = model_hf[f"{delta_pfx}.self_att.self_attention.attention_out_lora.lora_A.weight"].contiguous()
            out[f"{peft_pfx}.self_attn.o_proj.lora_B.weight"] = model_hf[f"{delta_pfx}.self_att.self_attention.attention_out_lora.lora_B.weight"].contiguous()

            out[f"{peft_pfx}.mlp.gate_proj.lora_A.weight"] = model_hf[f"{delta_pfx}.ffn.ffn.w_in.w_0_lora.lora_A.weight"].contiguous()
            out[f"{peft_pfx}.mlp.gate_proj.lora_B.weight"] = model_hf[f"{delta_pfx}.ffn.ffn.w_in.w_0_lora.lora_B.weight"].contiguous()
            out[f"{peft_pfx}.mlp.up_proj.lora_A.weight"] = model_hf[f"{delta_pfx}.ffn.ffn.w_in.w_1_lora.lora_A.weight"].contiguous()
            out[f"{peft_pfx}.mlp.up_proj.lora_B.weight"] = model_hf[f"{delta_pfx}.ffn.ffn.w_in.w_1_lora.lora_B.weight"].contiguous()
            out[f"{peft_pfx}.mlp.down_proj.lora_A.weight"] = model_hf[f"{delta_pfx}.ffn.ffn.w_out_lora.lora_A.weight"].contiguous()
            out[f"{peft_pfx}.mlp.down_proj.lora_B.weight"] = model_hf[f"{delta_pfx}.ffn.ffn.w_out_lora.lora_B.weight"].contiguous()
        else:
            print(f"Unexpected direction{direction}")

        
        
 
    for key in out:
        out[key] = out[key].half().cpu()
    
    torch.save(out, os.path.join(output_model_path, output_name))

# def transfer_malign_lora_2_peft_mul(input_model_path, output_model_path):
#     input_path = input_model_path
#     filter_only_lora(input_path, tmp_path)
#     transfer_malign_lora_2_peft(tmp_path, output_model_path,"peft")
    
parser = argparse.ArgumentParser(description='transfer malign lora to peft')
parser.add_argument('--input_model_path', type=str, help='input model path')
parser.add_argument('--output_model_path', type=str, help='output model path')
parser.add_argument("--direction", type=str, help='transform direction')
args = parser.parse_args()


# input_list=[args.input_model_path]
# output_list=[args.output_model_path]
# for input,output in zip(input_list,output_list):
#     # transfer_malign_lora_2_peft_mul(input, output)
transfer_malign_lora_2_peft(args.input_model_path, args.output_model_path, args.direction)
