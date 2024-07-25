# Stable Diffusion 3 (SD3) Training

## Overview

This repository contains the code for training Stable Diffusion 3 (SD3) model using the ComfyUI interface and the TotoroUI patcher.

## Requirements

* Python 3.8+
* PyTorch 1.12+
* TorchSDE, Einops, Diffusers, Accelerate, and Xformers libraries
* aria2 for downloading model weights

## Installation

1. Clone this repository: `git clone -b totoro2 https://github.com/camenduru/ComfyUI /content/TotoroUI`
2. Install required libraries: `pip install -q torchsde einops diffusers accelerate xformers==0.0.26.post1`
3. Install aria2: `apt -y install -qq aria2`

## Model Weights

Download the pre-trained SD3 model weights using aria2:
```bash
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/adamo1139/stable-diffusion-3-medium-ungated/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors -d /content/TotoroUI/model -o sd3_medium_incl_clips_t5xxlfp8.safetensors
