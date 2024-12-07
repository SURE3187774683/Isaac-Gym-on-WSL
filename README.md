# Humanioid(Windows+Wsl2(Ubuntu20.04)+MobaXterm)

# 1 ISAAC_GYM环境构建——驱动安装和torch配置

## 1.1 检查CUDA版本和是否配置成功
### import torch

### print(torch.__version__)  # 打印PyTorch版本
### print(torch.cuda.is_available())  # 检查CUDA是否可用
### print(torch.version.cuda)  # 打印CUDA版本（当前版本pytorch兼容的cuda）

## 1.2 配置版本：CUDA+Nvidia_Driver+pytorch

### CUDA Toolkit——11.6（！选择wsl版本）
https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

### NVIDIA Windows Driver_550（驱动可以向下兼容）
https://www.nvidia.com/en-us/drivers/

### Python_3.7.12
Isaac_gym创建的环境rlgpu的python版本是确定的,不能更改

### Pytorch_1.13.0+torchvision+torchaudio(NO pytorch-cuda)(！不要再安装CUDA覆盖了之前的版本)
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch -c nvidia
(https://pytorch.org/get-started/previous-versions/)


# 2 Some BUG
## 2.1 Segmentation fault (core dumped)
a.安装：sudo apt install vulkan-tools
检查：vulkaninfo

b.如果进入/usr/share/vulkan/icd.d查看文件信息只有三个文件则执行：

sudo add-apt-repository ppa:kisak/kisak-mesa

sudo apt update

sudo apt upgrade

c.如果vulkaninfo报错（symbol lookup error: /lib/x86_64-linux-gnu/libwayland-client.so.0: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.0）

检查依赖关系&&更新库文件：确保您的系统安装了所有必要的依赖库，特别是libffi
sudo apt-get install libffi-dev && sudo apt-get update && sudo apt-get upgrade

创建符号链接：如果上述步骤没有解决问题，您可以尝试创建一个符号链接来解决版本冲突问题。根据您的系统情况，您可能需要创建一个指向正确版本的libffi的符号链接。例如：
sudo ln -s /usr/lib/x86_64-linux-gnu/libffi.so.7 /usr/lib/x86_64-linux-gnu/libffi.so.6
请根据您的系统实际情况调整上述命令中的库文件版本

设置LD_LIBRARY_PATH：export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

# 3 bashrc文件配置
## # LIB
export LD_LIBRARY_PATH=/home/sure/miniconda3/envs/rlgpu/lib
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/lib/wsl/lib

## # xming
export DISPLAY=10.79.201.2:0.0

## # CUDA path
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.6


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sure/RL-code/humanplus-main/HST/isaacgym/python/isaacgym/_bindings/linux-x86_64

## # 设置cuda的编号
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

## # 删除vulkaninfo报错
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
