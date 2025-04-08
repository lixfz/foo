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


#SBATCH --job-name=llama2-test        # name
#SBATCH --nodes=2                  # nodes
#SBATCH --nodelist=ks-gpu-[4,5]
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:8                 # number of gpus per node
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --time=UNLIMITED              # maximum execution time (HH:MM:SS)

export GPUS_PER_NODE=8
export MASTER_ADDR=$DIST_INIT_ADDR
export MASTER_PORT=9901

export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_FILE=/slurmhome/aps/xuzhihui-test-nccl.%h.%p
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ib7s
export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,ib0,ib1,ib2,ib3
export OMPI_MCA_btl_base_exclude=openib
#export CUDA_LAUNCH_BLOCKING=1
# unset SLURM_TIMELIMIT


# srun --jobid $SLURM_JOBID bash -c \
python3 -m torch.distributed.run --nproc_per_node 8 --nnodes 6 --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT /root/test_network/train_model_pt.py \
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
    --seed 2023 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --gradient_accumulation_steps 1 \
    --output_dir "./result_llama2_1g_zh_en" \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --ddp_find_unused_parameters False \
    --overwrite_output_dir \
    --bf16 True
    
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
    
