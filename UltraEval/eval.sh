
# Your hf model path
HF_MODEL_NAME=/path/to/your/llama-2-7b-hf
NUMBER_OF_THREAD=1  # per-proc-gpus 
OUTPUT_BASE_PATH=./eval_results
mkdir -p $OUTPUT_BASE_PATH

# Change the following params to do different evaluations.
gpuid=$1 #"0"
port=$2 #5008
language=$3 # zh
task=$4 #math
GATE_PATH=$5 #"../checkpoints/Gates/zh_math.pt"


eval_batch_size=8 # Change the batch_size according to your device memory

## change test to dev to implement dev set evaluation.
CONFIG_PATH="configs/${language}_${task}_test.json"
# GATE_PATH="../checkpoints/Gates/${language}_${task}.pt" # Use gate from hf repo
LANGUAGE_LORA_PATH="../checkpoints/LoRAs/${language}_lora"
TASK_LORA_PATH="../checkpoints/LoRAs/${task}_lora"

URL="http://127.0.0.1:${port}/url-infer"

echo "Starting server on port $port with gate $GATE_PATH and language model $LANGUAGE_LORA_PATH and task model $TASK_LORA_PATH"

python URLs/url.py \
  --gpuid $gpuid  \
  --port $port  \
  --model_name $HF_MODEL_NAME \
  --gate_path $GATE_PATH \
  --language_lora_path $LANGUAGE_LORA_PATH \
  --task_lora_path $TASK_LORA_PATH &

sleep 30

python main.py \
    --model general \
    --model_args url=$URL,concurrency=$NUMBER_OF_THREAD \
    --config_path $CONFIG_PATH \
    --output_base_path $OUTPUT_BASE_PATH \
    --batch_size $eval_batch_size \
    --postprocess general_torch \
    --task ${language}_${task} \
    --write_out \

sleep 30
## get the PID of url.py
PID=$(ps -ef | grep "python URLs/url.py --gpuid $gpuid --port $port" | grep -v grep | awk '{print $2}')
## kill the process to release CUDA momery after evalution.
kill -9 $PID