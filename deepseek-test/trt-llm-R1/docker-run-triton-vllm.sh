#docker run -d --name t2 --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v /userdata:/userdata trt:2702 sleep infinity
#docker run -d --name ta --network host --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v /userdata:/userdata datacanvas/nctest:250328 sleep infinity
docker run  --name triton -itd \
	--network host \
	--ipc host \
	--shm-size=32G \
	--privileged \
	--ulimit nofile=65535:65535 \
	--gpus all -e NVIDIA_VISIBLE_DEVICES=all \
	-v /userdata:/userdata \
	nvcr.io/nvidia/tritonserver:25.02-vllm-python-py3 \
	bash

