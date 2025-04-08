  trtllm-bench \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --model_path  /userdata/llms/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  throughput \
  --backend pytorch \
  --max_batch_size 161 \
  --max_num_tokens 1160 \
  --dataset /userdata/dataset.txt \
  --tp 1 \
  --ep 1 \
  --pp 1 \
  --concurrency 1024 \
  --streaming \
  --kv_cache_free_gpu_mem_fraction 0.85 \
  --extra_llm_api_options /userdata/deepseek-test/trt-llm/extra-llm-api-config.yml 2>&1 | tee /userdata/trt_bench.log


  #   --engine_dir /userdata/llms/DeepSeek-R1-FP4 \