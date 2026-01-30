# Isaac GR00T Installation Guide for DGX Spark / NVIDIA Thor (ARM64)


This guide describes how to set up the environment for Isaac GR00T on **ARM-based NVIDIA platforms** (e.g., DGX Spark, NVIDIA Thor, Jetson Orin) where standard installation fails due to PyTorch and Flash Attention version mismatches.


> **Credit:** This guide is based on the solution provided by [@adityabhas22](https://github.com/adityabhas22) in [NVIDIA/Isaac-GR00T Issue #474](https://github.com/NVIDIA/Isaac-GR00T/issues/474).


## System Environment
- **OS:** Linux (Ubuntu 24.04)
- **Architecture:** aarch64 (ARM64)
- **CUDA:** 13.0
- **Python:** 3.10


## Step-by-Step Installation


### 1. Create Conda Environment
```
conda create -n gr00t python=3.10 -y
conda activate gr00t
pip install --upgrade pip setuptools wheel
```


### 2. Install PyTorch (CUDA 13.0)
The standard installation might install PyTorch 2.10, which causes issues with Flash Attention wheels meant for 2.9. However, we use a compatible wheel for 2.10 below.


# Check your current torch version if already installed

```
python -c "import torch; print(torch.__version__)"
```


### 3. Install Flash Attention (Pre-built for aarch64)
Since there is no official wheel for aarch64, use the pre-built wheel compatible with PyTorch 2.10 and CUDA 13.0.


# Remove existing version if any
```
pip uninstall flash-attn -y
```


# Install the specific wheel (v0.7.16)
```
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.10-cp310-cp310-linux_aarch64.whl
```

Credit to @mjun0812 for maintaining these wheels.



### 4. Install GR00T Package


Important: Install in editable mode without dependencies first to prevent overwriting the correct PyTorch version.


# Run this from the root of the repository


```
pip install -e . --no-deps
```


Then, manually install other dependencies if needed:
```
pip install albumentations==1.4.18 av==12.3.0 blessings==1.7 dm_tree==0.1.8 \
  einops==0.8.1 gymnasium==1.0.0 h5py==3.12.1 hydra-core==1.3.2 imageio==2.34.2 \
  kornia==0.7.4 matplotlib==3.10.0 "numpy>=1.23.5,<2.0.0" numpydantic==1.6.7 \
  omegaconf==2.3.0 opencv_python_headless==4.11.0.86 pandas==2.2.3 pydantic==2.10.6 \
  PyYAML==6.0.2 ray==2.40.0 Requests==2.32.3 tianshou==0.5.1 timm==1.0.14 \
  tqdm==4.67.1 transformers==4.51.3 typing_extensions==4.12.2 pyarrow==14.0.1 \
  wandb==0.18.0 fastparquet==2024.11.0 accelerate==1.2.1 peft==0.17.0 \
  protobuf==4.25.1 onnx==1.18.0 tyro pytest diffusers==0.30.2 pyzmq
```


### 5. Verification


Run the following script to verify the installation:


```
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

from flash_attn import flash_attn_func
print('Flash-attention: OK')
```


### 6. Training


For training, We use torchvision_av as the video backend since it uses pyAV which works on aarch64:

```
python scripts/gr00t_finetune.py \
  --dataset-path data/your_dataset \
  --output-dir ./checkpoints \
  --data-config so101_tricam_bimanual \
  --embodiment-tag new_embodiment \
  --num-gpus 1 \
  --max-steps 50000 \
  --batch-size 32 \
  --save-steps 5000 \
  --dataloader-num-workers 14 \
  --video-backend torchvision_av
```



### 7. Summary


| Package | Version | Notes |
| :--- | :--- | :--- |
| **Python** | 3.10 | Required for flash-attn wheel |
| **PyTorch** | 2.10.0+cu130 | From pytorch.org/whl/cu130 |
| **Flash-Attention** | 2.8.3 | From mjun0812 prebuilt wheels |
| **pytorch3d** | latest | Built from source |
| **Video Backend** | torchvision_av | Uses pyAV, works on aarch64 |
