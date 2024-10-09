
### 1. Download LoRA ckpts and gates(optional).

# export HF_ENDPOINT=https://hf-mirror.com

mkdir -p ./checkpoints

huggingface-cli download --token xxx --resume-download Bowen232/LoRA-Flow --local-dir ./checkpoints


### 2. Transform lora to bmversion. 

mkdir -p ./checkpoints/BM_LoRAs

LoRA_names=("zh" "es" "ru" "math" "code")

for LoRA_name in "${LoRA_names[@]}"; do
    python scripts/transform_lora.py \
        --input_model_path "./checkpoints/LoRAs/${LoRA_name}_lora" \
        --output_model_path "./checkpoints/BM_LoRAs/${LoRA_name}" \
        --direction "peft2malign" && 

    echo "Finishing transform ${LoRA_name} lora to bmtrain format."

done


### 3. Transform llama2-7b-hf to llama2-7b-mc
PATH_TO_LLAMA2_7B_HF="/path/to/llama2-7b-hf" 

python scripts/transform_llama_to_bmtrain.py $PATH_TO_LLAMA2_7B_HF "./checkpoints/Llama-2-7b-mc" &&
echo "Finishing transform llama2-7b-hf to bmtrain format."