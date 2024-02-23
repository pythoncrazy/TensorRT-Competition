# TensorRT-Competition
Pinak Paliwal's submission to the NVIDIA TensorRT competition


Steps to Reproduce:
- General Notes: I used Miniconda (conda 23.11.0) on Windows 11 (22H2, build version: 22621.3007). For docker, I used the wsl2 backend version of docker (25.0.3, build 4debf41, Docker Desktop version 4.27.2)
1. I ended up using docker via WSL2, which allowed me to use the GPU, and have an easy time running things. Additionally, I gained access to using TensorRT using docker, which was a big benefit, both speed-wise, and ease of use-wise.
2. Once I had WSL2 set up with Ubuntu 22.04, I ran the following command: `docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --dns 1.1.1.1 -v ./workspace:/workspace nvcr.io/nvidia/pytorch:23.11-py3 /bin/bash`


