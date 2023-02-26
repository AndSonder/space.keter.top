# 服务器上配置 Clash

在 Clash release 页面下载相应的版本，对于 Ubuntu 一般使用 clash-linux-amd64-vX.X.X.gz 版本：

```bash
wget https://github.com/Dreamacro/clash/releases/download/v1.10.0/clash-linux-amd64-v1.10.0.gz
```

:::tip

如果直接 wget 速度较慢的话，可以本地下载完成后，使用 SFTP 上传到 
Linux 服务器。

:::

然后使用 gunzip 命令解压，并重命名为 clash：

```bash

gunzip clash-linux-amd64-v1.10.0.gz
mv clash-linux-amd64-v1.10.0 clash

```

为 clash 添加可执行权限：

```bash
chmod u+x clash
```

Clash 运行时需要 Country.mmdb 文件，当第一次启动 Clash 时（使用 ./clash 命令） 会自动下载（会下载至 /home/XXX/.config/clash 文件夹下）。自动下载可能会因网络原因较慢，可以访问该[链接](https://github.com/Dreamacro/maxmind-geoip/releases)手动下载。

:::tip

Country.mmdb 文件利用 GeoIP2 服务能识别互联网用户的地点位置，以供规则分流时使用。

:::

## 配置文件

一般的网络服务提供了 Clash 订阅链接，可以直接下载链接指向的文件内容，保存到 config.yaml 中。或者使用订阅转换服务, 将其它订阅转换为 Clash 订阅。

这里推荐使用订阅转换服务，转换后的配置文件已添加更为强大的分流规则。就可以将 Clash 一直保持后台运行，自动分流，且会自动选择最优节点。

## Clash as a daemon

将 Clash 转变为系统服务，从而使得 Clash 实现常驻后台运行、开机自启动等。

```tip

普通用户需要 sudo 权限。

```

### 配置 systemd 服务

Linux 系统使用 systemd 作为启动服务器管理机制，首先把 Clash 可执行文件拷贝到 /usr/local/bin 目录，相关配置拷贝到 /etc/clash 目录。

```bash
sudo mkdir /etc/clash
sudo cp clash /usr/local/bin
sudo cp config.yaml /etc/clash/
sudo cp Country.mmdb /etc/clash/
```

创建 systemd 服务配置文件 sudo vim /etc/systemd/system/clash.service：

```bash
[Unit]
Description=Clash daemon, A rule-based proxy in Go.
After=network.target

[Service]
Type=simple
Restart=always
ExecStart=/usr/local/bin/clash -d /etc/clash

[Install]
WantedBy=multi-user.target
```

### 使用 systemctl

使用以下命令，让 Clash 开机自启动：

```bash
sudo systemctl enable clash
```

然后开启 Clash：

```bash
sudo systemctl start clash
```

### 查看 Clash 日志：

```bash
sudo systemctl status clash
sudo journalctl -xe
```

## 使用代理

安装 proxychains4：

```bash
sudo apt-get install proxychains4
```

编辑配置文件：

```bash
sudo vim /etc/proxychains4.conf
```

修改为

```bash
socks4 127.0.0.1 7890
```

## 测试

对于想要使用代理的命令，使用 proxychains4 前缀即可：

```bash
proxychains4 curl www.google.com
```





