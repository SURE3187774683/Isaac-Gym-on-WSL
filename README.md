# Humanioid(Windows+Wsl2(Ubuntu20.04)+MobaXterm)

# 1 ISAAC_GYM环境构建——驱动安装和torch配置

## 1.1 检查CUDA版本和pytorch版本
import torch

print(torch.__version__)  # 打印PyTorch版本
print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.version.cuda)  # 打印CUDA版本（当前版本pytorch兼容的cuda）

## 1.2 配置版本：CUDA_11.6+Nvidia_Driver_550+python_3.7.12+pytorch_1.13.0

### CUDA Toolkit（！选择wsl版本）
https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

### NVIDIA Windows Driver（550）（驱动可以向下兼容）
https://www.nvidia.com/en-us/drivers/

### Isaac_gym创建的环境rlgpu的python版本是确定的（3.7.12）

### 安装torch（！不要再安装CUDA覆盖了之前的版本）（官方指令）
(https://pytorch.org/get-started/previous-versions/)


# 2.Some BUG
## 2.1
