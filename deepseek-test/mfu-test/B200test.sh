#!/bin/bash


export GPUS_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=9901

export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_FILE=/slurmhome/aps/xuzhihui-test-nccl.%h.%p
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_HCA=ib7s


python3 -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT train_model_pt.py \
    --config_name "Qwen2.5-72B-Instruct" \
    --tokenizer_name_or_path "llama2_chinese_merged_tokenizer" \
    --deepspeed ds3.json \
    --save_safetensors False \
    --validation_split_percentage 0.005 \
    --dataloader_num_workers 16 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --warmup_steps 5 \
    --seed 2023 \
    --num_train_epochs 10000 \
    --lr_scheduler_type cosine \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --logging_strategy steps \
    --logging_steps 1 \
    --gradient_accumulation_steps 1 \
    --output_dir "./result_llama2_1g_zh_en" \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --ddp_find_unused_parameters False \
    --overwrite_output_dir \
    --bf16 True \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1
    