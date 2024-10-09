#! /usr/bin/env python3
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import bmtrain as bmt
from functools import partial
import time
import os
import sys
sys.path.append("./train/ModelCenter")
sys.path.append("./train")
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
import random
import math
from dataset import PromptIterableDataset, collator, load_train_dataset
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def load_lora_weight(root_path,lora_name):
    lora_weight_dict = torch.load(os.path.join(root_path,lora_name+"/lora.pt"))
    keys_to_update = [key for key in lora_weight_dict.keys()]
    
    if lora_name == "ru" or lora_name == "es":
        lora_name = "zh"  ## TODO: 让lora_name在运行过程中是对应的language, task，不过这并没有本质影响
    
    if lora_name == "code":
        lora_name = "math"
        
    for key in keys_to_update:
        ###向key中添加lora_name
        new_key = key.split("lora_")[0]+lora_name+"."+"lora_"+key.split("lora_")[1]
        ###更换成新key
        lora_weight_dict[new_key] = lora_weight_dict.pop(key)
    return lora_weight_dict



def get_model_tokenizer(args):
    bmt.print_rank("loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    bmt.print_rank("finished")
    bmt.print_rank("loading model...")
    model = Llama.from_pretrained(args.model_name_or_path)
    bmt.init_parameters(model)
    model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.pt"),strict=False)
    #load trained LoRA modules
    lora_list = args.lora_list.split(',')
    for lora in lora_list:
        lora_weight_dict = load_lora_weight(args.lora_root_path,lora)
        model.load_state_dict(lora_weight_dict,strict=False)
    for n,p in model.named_parameters():
        if "lora_fusion_gate" in n or "weight_bias" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    bmt.print_rank("finished")

    tokenizer.pad_token = tokenizer.eos_token
 
    return model, tokenizer

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), weight_decay=args.weight_decay, eps=1e-5, betas=(0.9, 0.95)
    )

    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    if args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    elif args.lr_decay_style == "cosine":
        bmt.print_rank("use cosine")
        lr_scheduler = bmt.lr_scheduler.Cosine(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    elif args.lr_decay_style == "noam":
        bmt.print_rank("use noam")
        lr_scheduler = bmt.lr_scheduler.Noam(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    else:
        raise NotImplementedError
    return lr_scheduler


def setup_model_and_optimizer(args):
    model, tokenizer = get_model_tokenizer(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler



def train(args):

    bmt.init_distributed(
        seed=args.seed,
        zero_level=3,
    )
    

    
    original_dataset = []
    if args.train_data_path is not None:
        original_dataset += load_train_dataset(args.train_data_path)

    random.shuffle(original_dataset)
    bmt.print_rank("total training instance number:", len(original_dataset))
    scaling = 2 if "ru,code" in args.lora_list else 1
    
    args.train_iters = int((args.epochs * len(original_dataset)) / (args.batch_size_per_device * bmt.world_size())) + 1
    args.warmup_iters = int(args.train_iters * args.warmup_ratio * scaling)
    if args.lr_decay_style == 'noam':
        args.lr = args.lr * math.sqrt(args.warmup_iters)
    
     
    bmt.print_rank("total training iterations:", args.train_iters)
    bmt.print_rank("warm-up iterations:", args.warmup_iters)

    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale)
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    
    bmt.synchronize()
    
    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())

    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()
    train_start_time = time.time()
    global_step = 0

    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    
    def load_and_save_trainable_params(model_file_path):
        state_dict = torch.load(model_file_path, map_location='cpu')
        
        gate_params = {k: v.clone().contiguous() for k, v in state_dict.items() if "lora_fusion_gate.weight" in k or "weight_bias" in k}
        dir_name = os.path.dirname(model_file_path)
        gate_file_name = f"{args.lora_list.replace(',', '_')}.pt"
        gate_file_path = os.path.join(dir_name, gate_file_name)

        torch.save(gate_params, gate_file_path)

        os.remove(model_file_path)

        print(f"Gate parameters saved to {gate_file_path} and full model ckpt {model_file_path} deleted.")

    
    for epoch in range(args.epochs):
        epoch=epoch+1
        indices = torch.randperm(len(original_dataset))
        dataset = [original_dataset[i] for i in indices]
        
        data_per_gpu = len(dataset) // bmt.world_size()
        dataset = dataset[bmt.rank() * data_per_gpu : (bmt.rank() + 1) * data_per_gpu]


        dataset = PromptIterableDataset(dataset, tokenizer = tokenizer, max_seq_length = args.max_seq_length, teacher_forcing=True, truncate_method="tail",system_prompt = args.system_prompt)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_per_device, collate_fn=partial(collator, tokenizer))

        if global_step >= args.train_iters:
            break
        # progress_bar = tqdm(range(len(dataloader)), disable=not bmt.rank()==0, desc=f"epoch {epoch}")

        for step, inputs in enumerate(dataloader):
            if global_step < args.start_step:
                global_step += 1
                continue
            st = time.time()

            with bmt.inspect.inspect_tensor() as inspector:
                ids = inputs.pop("ids")
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
                labels = inputs.pop("labels")
                logits = model(**inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, len(tokenizer))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_func(shift_logits, shift_labels)
            
                global_loss = bmt.sum_loss(loss).item()

                optim_manager.backward(loss)

                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=args.clip_grad)
                    optim_manager.step()
                    optim_manager.zero_grad()
     
            global_step += 1
            iteration_time = time.time() - st

            avg_time_recorder.record(iteration_time)
            avg_loss_recorder.record(global_loss)

            # print time and loss
            if global_step % args.logging_step == 0:
                bmt.print_rank(
                    "| Epoch: {:3d} | Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} | time: {:.4f} seconds | total_time_passed: {:.4f} minutes".format(
                        epoch,
                        global_step,
                        global_loss,
                        avg_loss_recorder.value,
                        lr_scheduler.current_lr,
                        avg_time_recorder.value,
                        (time.time() - train_start_time) / 60
                    )
                )



            # save model
            if global_step % args.save_step == 0:

                save_dir = os.path.join(args.save_dir, f"checkpoints/step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                bmt.save(model, os.path.join(save_dir, "pytorch_model.pt"))

                if bmt.rank() == 0:
                    load_and_save_trainable_params(os.path.join(save_dir, "pytorch_model.pt"))

                bmt.print_rank(f"fusion gate saved at {save_dir}")
            
            if global_step == args.train_iters:
                break
        
        # save the model at the current epoch
        save_dir = os.path.join(args.save_dir, f"checkpoints/epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        bmt.save(model, os.path.join(save_dir, "pytorch_model.pt"))

        if bmt.rank() == 0:
            load_and_save_trainable_params(os.path.join(save_dir, "pytorch_model.pt"))

        bmt.print_rank(f"fusion gate saved at {save_dir}")

  

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--system_prompt", default=None, type=str)
    parser.add_argument("--lora_list", default=None, type=str)
    parser.add_argument("--lora_root_path", default=None, type=str)
    parser.add_argument("--train_data_path", default=None, type=str)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_name_or_path", default='/path/to/llama2-7b-mc')
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--max_seq_length", default=4096, type=int)
    parser.add_argument("--batch_size_per_device", default=2, type=int)
    parser.add_argument("--logging_step", default=1, type=int)
    parser.add_argument("--save_step", default=50000, type=int)
    parser.add_argument("--data_dir", default=None, type=str)
    
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--clip_grad", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay rate")
    parser.add_argument("--loss_scale", type=float, default=6553600, help="loss scale")
    parser.add_argument("--train_iters", type=int, default=2000000)
    parser.add_argument("--save_dir", type=str, default="/path/to/your/output_dir/")

    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument(
        "--lr-decay-style",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine", "exponential", "noam"],
        help="learning rate decay function",
    )
    parser.add_argument("--lr_decay_iters", type=int, default=None, help="lr decay steps")
    parser.add_argument(
        "--start_step", type=int, default=0, help="step to start or continue training"
    )

    args = parser.parse_args()
    if args.system_prompt is not None:
        args.system_prompt = args.system_prompt.replace('-',' ')
    else:
        import json
        with open('data/system_prompts.json', 'r') as f:
            systems_prompts = json.load(f) 
        system_prompt_key = args.lora_list.replace(',','_')
        args.system_prompt = systems_prompts.get(system_prompt_key)



    train(args)
    args.model = args.model_name_or_path.split("/")[-1]
