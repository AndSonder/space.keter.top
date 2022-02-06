# 重构UserAgent
:::tip
思考：网站如何来判定是人类正常访问还是爬虫程序访问？？？
:::

网站检测爬虫的一种最基本的方法就是查看请求头，这里我们有一个可以查看自己请求头的网站  **http://httpbin.org/get** ，如果用浏览器去浏览可以得到这样一个图片，UserAgent可以显示一些信息

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200304160350294.png)

然而当你用python去访问的时候：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200304160532726.png)

所以当我们使用爬虫去访问网页的时候，一定要伪装自己的请求头，否者一下就被识破了

## 重构请求头
 **urllib.request.Request**

- **作用**

创建请求对象(包装请求，重构User-Agent，使程序更像正常人类请求)

- **参数**

```
1、url：请求的URL地址
2、headers：添加请求头（爬虫和反爬虫斗争的第一步）
```

- **使用流程**

```
1、构造请求对象(重构User-Agent)
2、发请求获取响应对象(urlopen)
3、获取响应对象内容
```

这个被我叫做初学阶段的爬虫三板斧

- **如何伪造User-Agent**

很简单——百度啊，下面我列出了常用的User-Agent

```
User-Agent:的值
 
1) Chrome(谷歌)
Win7:
Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1
Win10:
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36
Chrome 17.0 – MAC
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11 
 
 
2) Firefox(火狐)
Win7:
Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0
Firefox 4.0.1 – MAC
Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1
 
 
3) Safari(Safari是苹果计算机的操作系统Mac OS中的浏览器)
safari 5.1 – MAC
Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50
safari 5.1 – Windows
Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50
 
 
4) Opera(欧朋浏览器可以在Windows、Mac和Linux三个操作系统平台上运行)
Opera 11.11 – MAC
User-Agent:Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11
Opera 11.11 – Windows
User-Agent:Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11
 
 
5) IE
IE 11
Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko
IE 9.0
Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;
IE 8.0
Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)
IE 7.0
Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)
IE 6.0
Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)
WinXP+ie8：
Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; GTB7.0)
WinXP+ie7：
Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)
WinXP+ie6：
Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)
 
6) 傲游
傲游（Maxthon）
User-Agent: Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)
 
 
7) 搜狗
搜狗浏览器 1.x
Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)
 
 
8) 360
User-Agent: Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)
 
9) QQ浏览器
User-Agent: Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)
```

- 示例

向测试网站（http://httpbin.org/get）发起请求，构造请求头并从响应中确认请求头信息

```python
from urllib import request

url = 'http://httpbin.org/get'
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 '
                        '(KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1'}
# 1. 创建请求对象-并未真正的发请求
req = request.Request(url,headers=headers)
# 2. 获取响应对象
res = request.urlopen(req)
# 3. 读取对象
html = res.read().decode('utf-8')
print(html)
```
