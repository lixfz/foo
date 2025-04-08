#!/bin/bash
max_concurrency_list=(100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
output_lengths=(8192)

echo "sharedgpt dataset benchmark"

for max_concurrency in "${max_concurrency_list[@]}"; do
  for output_length in "${output_lengths[@]}"; do
    num_prompts=5000
    cmd="python3 benchmark_serving.py --backend openai-chat --model /userdata/llms/deepseek-ai/DeepSeek-R1 \
      --endpoint /v1/chat/completions --num-prompts $num_prompts --max-concurrency $max_concurrency --dataset-path /userdata/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
      --sharegpt-output-len $output_length"
    echo $cmd
    eval "$cmd"
  done
done
