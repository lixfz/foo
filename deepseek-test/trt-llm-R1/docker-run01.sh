docker run -d --name t1 --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v /userdata:/userdata trt:01 sleep infinity
