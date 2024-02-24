# TensorRT-Competition
Pinak Paliwal's submission to the NVIDIA TensorRT competition


Steps to Reproduce:
- General Notes: I used Miniconda (conda 23.11.0) on Windows 11 (22H2, build version: 22621.3007). For the backend, I used wsl2, ubuntu 22.04, with miniforge. I used Python 3.10.0. My cuda version was 12.0.0.
- GPU Info:
    - I have a 2070 super, so 8gb of VRAM. I have tested on the above configuration and an RTX 3060 (8gb vram) as well.
1. I ended up using  WSL2, which allowed me to use the GPU, and have an easy time running things. I followed the instructions found here: https://github.com/NVIDIA/TensorRT/tree/release/8.6/demo/Diffusion, but modified them for running via wsl2. For reference, here are the commands I ran (initially starting in the backend directory in this repo):
- `git clone https://github.com/NVIDIA/TensorRT.git -b release/8.6 --single-branch`
- `cd TensorRT`
- `python3 -m pip install --upgrade pip`
- `python3 -m pip install --upgrade tensorrt`
- `export TRT_OSSPATH=/way/to/directory/root/backend/TensorRT/`
- `cd $TRT_OSSPATH/demo/Diffusion`
- `pip3 install -r requirements.txt`
- `pip3 install tensorrt`
- `export HF_TOKEN=<your access token>`
