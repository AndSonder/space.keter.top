---
title: 「白帽子学习笔记」加油NPT
date: 2021-05-26 14:01:02
tags: [课程学习,网络渗透测试]
categories: [课程学习,网络渗透测试]
katex: true
---



# 【白帽子学习笔记】加油NPT

> 考试复习内容，看到这篇博客的小伙伴要加油啊！冲冲冲！

## 0x01 PETS标准

整个渗透测试过程大致可以分为7个阶段：

- 前期与客户的交流阶段：确认是对目标的哪些设备和哪些问题进行测试，商讨过程中的主要因素有如下几个：
  - 渗透测试的目标
  - 进行渗透测试所需要的条件
  - 渗透测试过程中的限制条件
  - 渗透测试过程的工期
  - 渗透测试费用
  - 渗透测试过程的预期目标
- 情报的收集阶段：使用各种资源尽可能地获得要测试目标的相关信息
  - 被动扫描
  - 主动扫描
- 威胁建模阶段：这个阶段主要解释了如下的问题
  - 哪些资产所目标中的重要资产
  - 攻击时采用的技术和手段
  - 哪些群里可能会对目标系统造成攻击
  - 这些群体会使用哪些方法进行破坏
- 漏洞分析阶段
- 漏洞利用阶段
- 后渗透攻击阶段：尽可能地将目标被渗透后所可能产生的后果模拟出来，来给客户展示当前网络存在的问题会带来的风险
  - 控制权限的提升
  - 登录凭证的窃取
  - 重要信息的获取
  - 利用目标作为跳板
  - 建立长期的控制通道
- 报告阶段

<!--more-->

## 0x02 ethical hacking的意义



## 0x03 Kali基础

### 1x01 NAT和桥接的区别

**桥接模式：**在桥接模式下，VMWare虚拟出来的操作系统就像是局域网中的**一台独立的主机**（主机和虚拟机处于对等地位），**它可以访问网内任何一台机器**。在桥接模式下，我们往往需要为虚拟主机配置ＩＰ地址、子网掩码等（注意虚拟主机的iｐ地址要和主机ｉｐ地址在同一网段）。使用桥接模式的虚拟系统和主机的关系，就**如同连接在一个集线器上的两台电脑**；要让他们通讯就需要为虚拟系统配置ip地址和子网掩码。如果我们需要在局域网内建立一个虚拟服务器，并为局域网用户提供服务，那就要选择桥接模式。

**NAT：**是**Network Address Translation**的缩写，意即**网络地址转换**。使用NAT模式虚拟系统可把物理主机作为路由器访问互联网，NAT模式也是VMware创建虚拟机的默认网络连接模式。使用NAT模式网络连接时，VMware会在主机上建立单独的专用网络，用以在主机和虚拟机之间相互通信**。虚拟机向外部网络发送的请求数据'包裹'，都会交由NAT网络适配器加上'特殊标记'并以主机的名义转发出去**，外部网络返回的响应数据'包裹'，也**是先由主机接收，然后交由NAT网络适配器根据'特殊标记'进行识别并转发给对应的虚拟机**，因此，虚拟机在外部网络中不必具有自己的IP地址。**从外部网络来看，虚拟机和主机在共享一个IP地址，默认情况下，外部网络终端也无法访问到虚拟机。**此外，在一台主机上只允许有一个NAT模式的虚拟网络。因此，**同一台主机上的多个采用NAT模式网络连接的虚拟机也是可以相互访问的。**

### 1x02 基本操作

- ifconfig：查看IP信息
- netstat -r：查看网关

`如何判断两台主机时候在同一网段？`

将两台主机的IP分别与子网掩码进行与运算，比较运算结果是否相同；

## 0x04 被动扫描

### 1x01 什么是被动扫描？

主要指的是在目标无法察觉的情况下进行的信息收集

- 目标网站的所有者信息，例如：姓名、地址、电话、电子邮件等
- 目标网站的电子邮箱
- 目标网站的社交信息：QQ、微博、微信、论坛发帖等

### 1x02 zoomeye

ZoomEye是一款针对网络空间的搜索引擎，收录了互联网空间中的设备、网站及其使用的服务或组件等信息。

### 1x03 Google Hacking

- site ： 指定域名
- inurl：url存在关键字的网页
- intext：网页正文中的关键字
- filetype：指定文件类型
- intitle：网页标题中的关键字

## 0x05 主动扫描

主动扫描的范围要小得多。主动扫描一般都是针对目标发送特制的数据包，然后根据目标的反应来获得一些信息。这些信息主要包括目标主机是否在线、目标主机的指定端口是否开放、目标主机的操作系统、目标主机上运行的服务等。

### NMAP的应用

**扫描操作系统:**

`nmap -O IP`

**判断所在网络存活主机：**

扫描192.168.0.0/24网段上有哪些主机的存活的

`nmap -sP 192.168.0.0/24`

**扫描主机开放了哪些端口：**

TCP端口扫描：scan tcp

`nmap -sT IP`

UDP端口扫描：scan udp

`nmap -sU IP`

扫描全部端口：

`nmap-p "*" ip`

扫描前n的端口：

`nmap-top-ports n IP`

扫描指定的端口：

`nmap -P IP`

**扫描目标开启了哪些服务：**

`nmap -sV IP`

**将扫描结果保存为xml文件：**

`nmap -oX a.xml IP`

## 0x06 身份认证攻击

### 1x01 BurpSuite

主要需要知道一个作用就是Proxy：

拦截HTTP/S的代理服务器，作为一个在浏览器和目标应用程序之间的中间人，允许你拦截，查看，修改在两个方向上的原始数据流。

其他常用的功能还有：

- Spider(蜘蛛)——应用智能感应的网络爬虫，它能完整的枚举应用程序的内容和功能。
- Scanner(扫描器)——高级工具，执行后，它能自动地发现web 应用程序的安全漏洞。
- Intruder(入侵)——一个定制的高度可配置的工具，对web应用程序进行自动化攻击，如：枚举标识符，收集有用的数据，以及使用fuzzing 技术探测常规漏洞。
- Repeater(中继器)——一个靠手动操作来触发单独的HTTP 请求，并分析应用程序响应的工具。

### 如何设置安全的密码？

- 避开若口令
- 能记住的密码才是好密码
- 密码中包含数字，大小写英文
- 增加密码的长度
- 每个应用的密码都设置的具有一定差异

## 0x07 网络数据嗅探与欺骗

### 1x01 如何利于Wireshark恢复数据流中的文件

利用WireShark的包筛选去筛选出需要的包 ==> 跟踪数据流 ==> 找到需要的数据，选择原数据进行保存

常用的WireShark语句：

`tcp`：tcp流；

`http`：http数据流；

`http.request.method`：筛选HTTP数据流的请求方式；

`ip.src`：对于数据源地址进行筛选

`ip.dst`：对于目的地址筛选

### 1x02 arpspoof

开启端口转发，允许本机像路由器那样转发数据包

`echo 1 > /proc/sys/net/ipv4/ip_forward`

ARP投毒

`arpspoof -i eth0 -t IP1 IP2`(IP1是我们的攻击目标、IP2是网关IP地址)

-i eth0表示选择eth0这个网卡；

ARP攻击原理：

在局域网内的攻击方式主要有两种：

.<img src=" https://img-blog.csdn.net/20181013134230611?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RDMTI1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" style="zoom:67%;" />

(1) PC1：PC2不断的向PC1发送欺骗包，欺骗其为网关路由，最后导致PC1的ARP表遭到攻击；

(2) Route：PC2不断的向Route(网关路由)发送欺骗包，欺骗其为PC1；

`因为arp欺骗想把原理写明白需要很大的篇幅，这里就不细说了`

## 0x08 远程控制

### 1x01 正向连接和反向连接的区别

反向连接：攻击机设置一个端口（LPORT）和IP（LHOST），Payload在测试机执行连接攻击机IP的端口，这时如果在攻击机监听该端口会发现测试机已经连接。
正向连接：攻击机设置一个端口（LPORT），Payload在测试机执行打开该端口，以便攻击机可以接入。

### 1x02 反向连接的实施过程

攻击者先通过某个手段在目标机器上植入恶意代码，并且该代码可以被触法。攻击者设置一个端口和一个IP，当被攻击者执行了恶意代码后攻击者的机器就会获取被攻击者的代码。

## 0x09 漏洞扫描

### 1x01 工具们

工具：

- AWVS：漏洞扫描工具
- Beef：XSS漏洞利用工具
- SQLMAP：自动话sql注入工具：
  - 查询当前数据库：`sqlmap -u "IP" --cookie "xxx" --current-db`
  - 查询当前使用者：`--current user`
  - 爆破数据表：`-D xx --tables`
  - 爆破数据表表头：`-D xx -T xx --columns`
  - 爆破具体的列：`-D xx -T xx -C xx`

- Whatweb: 查询网页的基本信息 `whatweb IP`
- Wpscan:可以扫描WordPress中的多种安全漏洞
- Dirb：爆破用  `dirb -u http://IP`

- MeterSploit:功能非常强大的渗透工具

### 1x02 XSS攻击/SQL注入

**XSS攻击：**XSS攻击通常指的是通过利用[网页](https://baike.baidu.com/item/网页/99347)开发时留下的漏洞，通过巧妙的方法注入恶意指令代码到网页，使用户加载并执行攻击者恶意制造的网页程序。这些恶意网页程序通常是JavaScript，但实际上也可以包括Java、 VBScript、ActiveX、 Flash 或者甚至是普通的HTML。攻击成功后，攻击者可能得到包括但不限于更高的权限（如执行一些操作）、私密网页内容、会话和cookie等各种内容。

**SQL注入**攻击的原理：恶意用户在提交查询请求的过程中将SQL语句插入到请求内容中，同时程序本身对用户输入内容过分信任而未对恶意用户插入的SQL语句进行过滤，导致SQL语句直接被服务端执行。

## 0x10 OWASP TOP 10

1. [**Injection**](https://owasp.org/www-project-top-ten/2017/A1_2017-Injection). Injection flaws, such as SQL, NoSQL, OS, and LDAP injection, occur when untrusted data is sent to an interpreter as part of a command or query. The attacker’s hostile data can trick the interpreter into executing unintended commands or accessing data without proper authorization.
2. [**Broken Authentication**](https://owasp.org/www-project-top-ten/2017/A2_2017-Broken_Authentication). Application functions related to authentication and session management are often implemented incorrectly, allowing attackers to compromise passwords, keys, or session tokens, or to exploit other implementation flaws to assume other users’ identities temporarily or permanently.
3. [**Sensitive Data Exposure**](https://owasp.org/www-project-top-ten/2017/A3_2017-Sensitive_Data_Exposure). Many web applications and APIs do not properly protect sensitive data, such as financial, healthcare, and PII. Attackers may steal or modify such weakly protected data to conduct credit card fraud, identity theft, or other crimes. Sensitive data may be compromised without extra protection, such as encryption at rest or in transit, and requires special precautions when exchanged with the browser.
4. [**XML External Entities (XXE)**](https://owasp.org/www-project-top-ten/2017/A4_2017-XML_External_Entities_(XXE)). Many older or poorly configured XML processors evaluate external entity references within XML documents. External entities can be used to disclose internal files using the file URI handler, internal file shares, internal port scanning, remote code execution, and denial of service attacks.
5. [**Broken Access Control**](https://owasp.org/www-project-top-ten/2017/A5_2017-Broken_Access_Control). Restrictions on what authenticated users are allowed to do are often not properly enforced. Attackers can exploit these flaws to access unauthorized functionality and/or data, such as access other users’ accounts, view sensitive files, modify other users’ data, change access rights, etc.
6. [**Security Misconfiguration**](https://owasp.org/www-project-top-ten/2017/A6_2017-Security_Misconfiguration). Security misconfiguration is the most commonly seen issue. This is commonly a result of insecure default configurations, incomplete or ad hoc configurations, open cloud storage, misconfigured HTTP headers, and verbose error messages containing sensitive information. Not only must all operating systems, frameworks, libraries, and applications be securely configured, but they must be patched/upgraded in a timely fashion.
7. [**Cross-Site Scripting XSS**](https://owasp.org/www-project-top-ten/2017/A7_2017-Cross-Site_Scripting_(XSS)). XSS flaws occur whenever an application includes untrusted data in a new web page without proper validation or escaping, or updates an existing web page with user-supplied data using a browser API that can create HTML or JavaScript. XSS allows attackers to execute scripts in the victim’s browser which can hijack user sessions, deface web sites, or redirect the user to malicious sites.
8. [**Insecure Deserialization**](https://owasp.org/www-project-top-ten/2017/A8_2017-Insecure_Deserialization). Insecure deserialization often leads to remote code execution. Even if deserialization flaws do not result in remote code execution, they can be used to perform attacks, including replay attacks, injection attacks, and privilege escalation attacks.
9. [**Using Components with Known Vulnerabilities**](https://owasp.org/www-project-top-ten/2017/A9_2017-Using_Components_with_Known_Vulnerabilities). Components, such as libraries, frameworks, and other software modules, run with the same privileges as the application. If a vulnerable component is exploited, such an attack can facilitate serious data loss or server takeover. Applications and APIs using components with known vulnerabilities may undermine application defenses and enable various attacks and impacts.
10. [**Insufficient Logging & Monitoring**](https://owasp.org/www-project-top-ten/2017/A10_2017-Insufficient_Logging%26Monitoring). Insufficient logging and monitoring, coupled with missing or ineffective integration with incident response, allows attackers to further attack systems, maintain persistence, pivot to more systems, and tamper, extract, or destroy data. Most breach studies show time to detect a breach is over 200 days, typically detected by external parties rather than internal processes or monitoring.







































































































