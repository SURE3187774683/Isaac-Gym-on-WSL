Humanioid
# My Env:Windows+Wsl2(Ubuntu20.04)+MobaXterm

# 1.ISAAC_GYM环境构建——驱动安装和torch配置

## 检查CUDA版本和pytorch版本
import torch

print(torch.__version__)  # 打印PyTorch版本
print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.version.cuda)  # 打印CUDA版本（当前版本pytorch兼容的cuda）

## CUDA_12.1+Nvidia_Driver_550+python_3.8.10+pytorch_2.2.2

### CUDA Toolkit（！选择wsl版本）
https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

### NVIDIA Windows Driver（550）（驱动可以向下兼容）
https://www.nvidia.com/en-us/drivers/

### 安装python
conda install python==3.8.10

### 安装torch（！不要再安装CUDA覆盖了之前的版本）（官方指令）
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121


