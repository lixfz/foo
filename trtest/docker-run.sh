#docker run -d --name t2 --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v /userdata:/userdata trt:2702 sleep infinity
#docker run -d --name ta --network host --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v /userdata:/userdata datacanvas/nctest:250328 sleep infinity
docker run --rm --name ta3 -it \
	--network host \
	--ipc host \
	--shm-size=32G \
	--privileged \
	--ulimit nofile=65535:65535 \
	--ulimit stack=655350:655350 \
	--gpus all -e NVIDIA_VISIBLE_DEVICES=all \
	-v /userdata:/userdata \
	-w /userdata/foo/trtest \
	trt:250408 \
	bash

	#datacanvas/nctest:250328 \
	#trt:250401 \
