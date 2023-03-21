# Paddle TensorRT 环境配置

## 使用 Docker 安装 Paddle

首先 clone Paddle 的源码：

```bash
git clone https://github.com/PaddlePaddle/Paddle.git
```

然后 cd 进入 Paddle 目录，开始安装 Docker

### 安装 NVIDIA Docker

Paddle TensorRT 只能在 NVIDIA GPU 上运行，因此需要安装 NVIDIA Docker。

```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```
### 拉取Docker镜像

:::tip

拉取镜像，注意一定要拉取带有TensorRT的镜像，否则后续无法使用TensorRT。

:::

```bash
nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.7-cudnn8.4-trt8.4-gcc8.2
nvidia-docker run -d -p 8888:22 -v $PWD:/home/paddle --name paddle-trt paddle:latest-dev-cuda11.7-cudnn8.4-trt8.4-gcc8.2 /bin/bash
```

## 编译 Paddle 和 TensorRT

```bash
mkdir build & cd build

time cmake .. -DTENSORRT_ROOT=/usr -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_PYTHON=ON -DWITH_TENSORRT=ON 
time make -j$(nproc)

pip3.7 install python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl --force-reinstall

python3.7 /home/paddle/python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_elementwise.py
python3.7 /home/paddle/python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_pad3d.py
python3.7 /home/paddle/python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_temporal_shift.py
python3.7 /home/paddle/python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_pad.py
```

## sh 脚本

```bash
#!/bin/bash
time cmake .. -DTENSORRT_ROOT=/usr -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_PYTHON=ON -DWITH_TENSORRT=ON
chmod -R 777 ./
chmod -R 777 ../
time make -j$(nproc)
pip install python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl --force-reinstall
python /home/paddle/python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_pad3d.py
```