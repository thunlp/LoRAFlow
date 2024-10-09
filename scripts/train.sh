CODE=./train
EXP=./train/output
mkdir -p $EXP
LOGPATH=./scripts/logs
mkdir -p $LOGPATH
language=$1
task=$2
lr=$3
bsz=$4
MASTER_PORT=$5
FREQ=1 # change the gradient_accumulation_steps if needed


# language="es"
# task="math"

epoch=5
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

# 构建日志文件名
LOG_FILE="./scripts/logs/${language}_${task}_bsz_$((bsz * FREQ * NUM_GPUS))_lr_${lr}_epoch_${epoch}.log"

# 构建 OPTS
OPTS=""
OPTS+=" --gradient_accumulation_steps $FREQ"
OPTS+=" --model_name_or_path ./checkpoints/Llama-2-7b-mc"
OPTS+=" --save_dir ${EXP}/${language}_${task}_bsz_$((bsz * FREQ * NUM_GPUS))_lr_${lr}_epoch_${epoch}_7b"
OPTS+=" --lora_list ${language},${task}"
OPTS+=" --lora_root_path  ./checkpoints/BM_LoRAs"
OPTS+=" --train_data_path ./data/train/${language}_${task}_train.jsonl"
OPTS+=" --batch_size_per_device $bsz --epochs $epoch --lr $lr"


# 运行任务
torchrun --nnodes=1 --nproc_per_node=2 --master_port=$MASTER_PORT $CODE/train.py ${OPTS} &> $LOG_FILE
