# Env : Wsl2(Ubuntu20.04) + Cuda-12.4 + cuda_driver_556 + NVIDIA_3060

## Pipeline：

## 一、按照帖子安装Isaacgym

https://blog.csdn.net/m0_37802038/article/details/134629194?ops_request_misc=&request_id=&biz_id=102&utm_term=Isaacgym%E6%8A%A5%E9%94%99%20Segmentation%20fault&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-9-134629194.142^v100^pc_search_result_base6&spm=1018.2226.3001.4187

## 二、一些Bug
### 1. 运行报错：[Error] [carb] Failed to acquire interface: [carb::gym::Gym v0.1], by client: carb.gym.python.gym_38/gym_37 (plugin name: (null))

解决方法：
安装CUDA（自行查看应该安装的版本：https://blog.csdn.net/learner_jj/article/details/140122805）
(https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

### 2. 运行报错：“ImportError: libpython3.8.so.1.0/ libpython3.7.so.1.0: cannot open shared object file: No such file or directory”

解决方法：
export LD_LIBRARY_PATH=/path_to_your_env/lib:$LD_LIBRARY_PATH
e.g.: export LD_LIBRARY_PATH=/home/sure/miniconda3/envs/rlgpu/lib:$LD_LIBRARY_PATH

### 3. 运行报错：Warning: failed to preload CUDA lib internal error : libcuda.so!

解决方法：
首先which libcuda.so查找，然后将lib库添加到环境变量
e.g. export LD_LIBRARY_PATH=LD_LIBRARY_PATH:path

### 4. 黑屏
sudo apt-get update && sudo apt-get install ubuntu-desktop && reboot

### 5. 运行报错：Segmentation fault (core dumped)

解决方法：
#### 5.1  输入vulkaninfo看是否有一长串输出，如果没有，执行sudo apt install vulkan-tools

#### 5.2 cd /usr/share/vulkan/icd.d看文件数量，如果只有三个文件，执行
sudo add-apt-repository ppa:kisak/kisak-mesa &&
sudo apt update &&
sudo apt upgrade

#### 5.3 . 如果vulkaninfo报错（symbol lookup error: /lib/x86_64-linux-gnu/libwayland-client.so.0: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.0）
执行
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

或执行
sudo apt-get install libffi-dev && sudo apt-get update && sudo apt-get upgrade &&
sudo ln -s /usr/lib/x86_64-linux-gnu/libffi.so.7 /usr/lib/x86_64-linux-gnu/libffi.so.6 &&
sudo cp /home/sure/miniconda3/envs/rlgpu/lib/libpython3.7m.so.1.0 /usr/lib &&
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

#### 5.4 如果isaacgym环境下可以跑example，但hgym下没法跑，报错Segmentation fault (core dumped)
配置vulkan的路径：
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.x86_64.json

## 三、环境变量设置

### # cuda-12.4
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.4

### # display
export DISPLAY=localhost:0.0
### # isaacgym
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sure/workspace/isaacgym/python/isaacgym/_bindings/linux-x86_64
### # use cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/wsl/lib/
### # rlgpu环境可以跑example，但hgym没法跑，指定Vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.x86_64.json

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

## 四、有用的帖子
### 4.1 安装教程
https://blog.csdn.net/m0_37802038/article/details/134629194?ops_request_misc=&request_id=&biz_id=102&utm_term=Isaacgym%E6%8A%A5%E9%94%99%20Segmentation%20fault&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-9-134629194.142^v100^pc_search_result_base6&spm=1018.2226.3001.4187

https://blog.csdn.net/wsygbthhhh/article/details/143918730?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522a54726eeae0be86042a6003e0c40c814%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=a54726eeae0be86042a6003e0c40c814&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-6-143918730-null-null.142^v100^pc_search_result_base6&utm_term=isaacgym%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4187

https://blog.csdn.net/littlewells/article/details/140179837

https://blog.csdn.net/weixin_44061195/article/details/131830133
### 4.2 CUDA Python tookit pytorch匹配关系
https://pytorch.org/get-started/previous-versions/

https://blog.csdn.net/weixin_41809117/article/details/141246957

https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions

### 4.3 英伟达驱动卸载安装教程
https://blog.csdn.net/Perfect886/article/details/119109380
