---
title: 「mist」配置mist论文环境
date: 2021-05-10 14:01:02
tags: mist
katex: true
---

个人配置mistGPU论文训练环境的代码：

```shell
cp /data/*.zip ./
unzip env.zip
mv torch-1.7.1+cu101-cp36-cp36m-linux_x86_64\ .whl torch-1.7.1+cu101-cp36-cp36m-linux_x86_64.whl
unzip detectron2.zip
conda create -n "corona2" python=3.6
conda install cudatoolkit=11.1
pip3 install torch-1.7.1+cu101-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision-0.8.2+cu101-cp36-cp36m-linux_x86_64.whl
pip3 install detectron2-0.
3+cu101-cp36-cp36m-linux_x86_64.whl
pip3 install opencv-python
cd cocoapi/PythonAPI
python3 setup.py build_ext --inplace --user
python3 setup.py install --user
```



cuda11.0的版本配置：

```shell
conda create -n "corona2" python=3.7
conda install cudatoolkit=11.0
pip3 install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip3 install opencv-python
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

