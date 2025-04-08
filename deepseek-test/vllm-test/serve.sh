# export VLLM_TRACE_FUNCTION=1

# vllm serve /userdata/llms/deepseek-ai/DeepSeek-R1 -tp 8 --trust-remote-code

vllm serve /userdata/llms/deepseek-ai/DeepSeek-R1 -tp 8 --trust-remote-code \
    --max-model-len 65536  --enable-reasoning --reasoning-parser deepseek_r1 --disable-log-requests
