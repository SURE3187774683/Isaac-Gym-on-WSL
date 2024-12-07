### humanioid

# 1.ISAAC_GYM环境构建——驱动安装和torch配置

## 检查CUDA版本和pytorch版本
import torch

print(torch.__version__)  # 打印PyTorch版本
print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.version.cuda)  # 打印CUDA版本（当前版本pytorch兼容的cuda）

## CUDA 12.1+cudnn550+python3.8（官方指令）

### CUDA Toolkit（！选择wsl版本）
https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

### 安装torch（！不要再安装CUDA覆盖了之前的版本）
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 -c pytorch -c nvidia
