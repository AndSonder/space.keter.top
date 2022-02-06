# request.get()参数详解

## 查询参数
- 参数类型
```
字典,字典中键值对作为查询参数
```
- 使用方法
```
1、res = requests.get(url,params=params,headers=headers)
2、特点: 
   * url为基准的url地址，不包含查询参数
   * 该方法会自动对params字典编码,然后和url拼接
```
- 事例
```
import requests
baseurl = 'http://tieba.baidu.com/f?'
params = {
  'kw' : '赵丽颖吧',
  'pn' : '50'
}
headers = {'User-Agent' : 'Mozilla/4.0'}
# 自动对params进行编码,然后自动和url进行拼接,去发请求
res = requests.get(url=baseurl,params=params,headers=headers)
res.encoding = 'utf-8'
print(res.text)
```

## Web客户端验证参数-auth
- 作用及类型
```
1、针对于需要web客户端用户名密码认证的网站
2、auth = ('username','password')
```
当然了大多数网站都有反爬虫系统的啦，就是类似于验证码啊之类的东西，想要登陆状态的话还是需要用cookie来保存自己的登陆状态

## SSL证书认证参数-verify
- 适用网站及场景
```
1、适用网站: https类型网站但是没有经过 证书认证机构 认证的网站
2、适用场景: 抛出 SSLError 异常则考虑使用此参数
```
- 参数类型
```
1、verify=True(默认)   : 检查证书认证
2、verify=False（常用）: 忽略证书认证
# 示例
response = requests.get(
	url=url,
	params=params,
	headers=headers,
	verify=False
)
```

## 代理参数-proxies
- 定义

```
1、定义: 代替你原来的IP地址去对接网络的IP地址。
2、作用: 隐藏自身真实IP,避免被封。
```

**普通代理**
- 获取代理IP网站

```
西刺代理、快代理、全网代理、代理精灵、... ...
```
- 参数类型
```
1、语法结构
   	proxies = {
       	'协议':'协议://IP:端口号'
   	}
2、示例
    proxies = {
    	'http':'http://IP:端口号',
    	'https':'https://IP:端口号'
	}
```

当然了这些网页里提供的代理IP大多数因为用的人太多了不能用，比如西刺代理里面的ip，用爬虫爬了1000多个下来能用的居然只有80多个，过两天一个又有好多不行了，一般大一些公司会给你提供专门的资金去购买代理IP，如果没有的话就好好的积累积累自己的IP池吧

- 示例
使用免费普通代理IP访问测试网站: http://httpbin.org/get

```
import requests

url = 'http://httpbin.org/get'
headers = {
    'User-Agent':'Mozilla/5.0'
}
# 定义代理,在代理IP网站中查找免费代理IP
proxies = {
    'http':'http://112.85.164.220:9999',
    'https':'https://112.85.164.220:9999'
}
html = requests.get(url,proxies=proxies,headers=headers,timeout=5).text
print(html)
```

你还可以自己写一个属于自己的代理IP接口直接放到python的环境变量路径里就可以直接用啦~