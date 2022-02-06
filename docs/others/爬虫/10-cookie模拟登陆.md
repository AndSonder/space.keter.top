# cookie模拟登陆

## 方法1 利用cookie
```
1、先登录成功1次,获取到携带登陆信息的Cookie（处理headers） 
2、利用处理的headers向URL地址发请求
```

## 方法2 利用requests.get()中cookies参数
```
1、先登录成功1次,获取到cookie,处理为字典
2、res=requests.get(xxx,cookies=cookies)
```
## 方法3  利用session会话保持
```
1、实例化session对象
      session = requests.session()
2、先post : session.post(post_url,data=post_data,headers=headers)
      1、登陆，找到POST地址: form -> action对应地址
      2、定义字典，创建session实例发送请求
         # 字典key ：<input>标签中name的值(email,password)
         # post_data = {'email':'','password':''}
3、再get : session.get(url,headers=headers)
```


