#!/bin/bash


export TORCH_CUDA_ARCH_LIST=10.0
model_path=/userdata/llms/DeepSeek-R1-FP4/
#model_path=/userdata/llms/deepseek-ai/DeepSeek-R1/

echo model_path: $model_path

trtllm-serve --host 0.0.0.0 --port 30000  \
    --backend pytorch  \
    --tp_size 8  --pp_size 1 --ep_size 4 \
    --max_batch_size 300 \
    --max_num_tokens  32768 \
    --max_seq_len 65536 \
    --trust_remote_code  \
    --kv_cache_free_gpu_memory_fraction 0.8 \
    $model_path

    #--tokenizer  /userdata/llms/meta-llama/Llama-3.1-8B-Instruct \
    #/userdata/llms/meta-llama/Llama-3.1-8B-Instruct-ckpt-trt-engine

    #--tp_size 8  --pp_size 1 --gpus_per_node 8 \
    #/userdata/llms/deepseek-ai/hf/DeepSeek-R1
    #/userdata/llms/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    #/userdata/llms/deepseek-ai/DeepSeek-V3/
