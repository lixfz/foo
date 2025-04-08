# docker run -d --name trt-llm -e NVIDIA_VISIBLE_DEVICES=all -p 8000:8000 -v /userdata:/userdata tensorrt_llm/release:latest sleep infinity

docker run -d --name sgl -it --network host --ipc host --shm-size=128G \
    --privileged \
    --ulimit nofile=65536:65536 \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v /userdata:/userdata \
    nvcr.io/nvidia/pytorch:25.03-py3 \
    bash
