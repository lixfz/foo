python vllm/benchmarks/benchmark_serving.py \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --model '/local/deepseek-ai/DeepSeek-R1' \
        --dataset-name sharegpt \
        --dataset-path "/capacity/vksdata/workshops/workshop-3e21f852-6461-4a62/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json" \
        --max-concurrency 100 \
        --num-prompts 3000  \
        --save-result