# Ubuntu 配置 fish 和 conda 环境


## 安装fish

```bash
# 安装最新版fish
sudo apt-add-repository ppa:fish-shell/release-3                                               
sudo apt update 	
sudo apt install fish
conda init fish 

fish
```

## 美化fish

```bash
# 安装 oh my fish
git clone https://gitee.com/mirrors/oh-my-fish
cd oh-my-fish
bin/install --offline
```

```bash
omf install bobthefish
```

