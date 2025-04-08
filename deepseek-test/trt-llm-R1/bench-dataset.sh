cd /userdata/TensorRT-LLM

python benchmarks/cpp/prepare_dataset.py \
  --tokenizer=/userdata/llms/DeepSeek-R1-FP4 \
  --stdout token-norm-dist \
  --num-requests=8192 \
  --input-mean=1000 \
  --output-mean=1000 \
  --input-stdev=0 \
  --output-stdev=0 > /userdata/dataset.txt