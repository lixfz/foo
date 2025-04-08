# docker run -d --name trt-llm -e NVIDIA_VISIBLE_DEVICES=all -p 8000:8000 -v /userdata:/userdata tensorrt_llm/release:latest sleep infinity

docker run -d --name triton -it --network host --ipc host --shm-size=32G \
    --privileged \
    --ulimit nofile=65536:65536 \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v /userdata:/userdata \
    nvcr.io/nvidia/tritonserver:25.02-vllm-python-py3 \
    bash