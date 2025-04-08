#!/bin/bash

export STATEFULSET_NAME=`echo $HOSTNAME | sed -r 's/(.*)-([0-9]+)$/\1/'`
export LEADER_HOSTNAME=`echo $HOSTNAME | sed -r 's/(.*-)([0-9]+)$/\10/'`
COMMAND="getent hosts $LEADER_HOSTNAME.$STATEFULSET_NAME"
echo $COMMAND
while ! $COMMAND; do
  echo "waiting for leader to bootstrap..."
  sleep 1
done

export DIST_INIT_ADDR=`getent hosts $LEADER_HOSTNAME.$STATEFULSET_NAME | awk -F ' ' '{print $1}'`
export RANK=`hostname |  sed -r 's/(.*-)([0-9]+)$/\2/'`
echo $DIST_INIT_ADDR
echo $RANK


export GPUS_PER_NODE=8
export MASTER_ADDR=$DIST_INIT_ADDR
export MASTER_PORT=9901

export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_FILE=/slurmhome/aps/xuzhihui-test-nccl.%h.%p
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ib7s
echo $NODES

cp /root/test_network/engine.py /opt/aps/python/lib/python3.10/site-packages/deepspeed/runtime/engine.py 
cp /root/test_network/profiler.py /opt/aps/python/lib/python3.10/site-packages/deepspeed/profiling/flops_profiler/profiler.py


python3 -m torch.distributed.run --nproc_per_node 8 --nnodes $NODES --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT /root/test_network/train_model_pt.py \
    --config_name "Qwen2.5-72B-Instruct" \
    --tokenizer_name_or_path "llama2_chinese_merged_tokenizer" \
    --deepspeed ds3.json \
    --save_safetensors False \
    --validation_split_percentage 0.005 \
    --dataloader_num_workers 16 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
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
    --save_steps 50 \
    --save_total_limit 3
    
    # --warmup_steps 30 \
    # --dataset_cache  "/slurmhome/aps/data/llama2/continue_train/zh/wiki_zh_1g_dataset_0.29b" \
    # --model_name_or_path "falcon-40b-instruct" \
    # --use_peft False \
    # --trainable "query_key_value" \
    # --lora_dropout 0.05 \
    # --lora_rank 8 \
    # --lora_alpha 32 \
    # --report_to none'

    # --fsdp "full_shard auto_wrap"  \
    # --fsdp_config fsdp.json \
    #  \
    # --report_to wandb
    
