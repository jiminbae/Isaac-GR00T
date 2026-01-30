# Isaac GR00T Installation Guide for DGX Spark / NVIDIA Thor (ARM64)

This guide describes how to set up the environment for Isaac GR00T on **ARM-based NVIDIA platforms** (e.g., DGX Spark, NVIDIA Thor, Jetson Orin) where standard installation fails due to PyTorch and Flash Attention version mismatches.

> **Credit:** This guide is based on the solution provided by [@adityabhas22](https://github.com/adityabhas22) in [NVIDIA/Isaac-GR00T Issue #474](https://github.com/NVIDIA/Isaac-GR00T/issues/474), and verified on NVIDIA Thor.

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
pip install [https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.1
```

### 4. Install GR00T Package

Important: Install in editable mode without dependencies first to prevent overwriting the correct PyTorch version.

# Run this from the root of the repository

```
pip install -e . --no-deps
```
Then, manually install other dependencies if needed.

5. Verification
Run the following script to verify the installation:

```
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

from flash_attn import flash_attn_func
print('Flash-attention: OK')
```
