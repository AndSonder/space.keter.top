# 🍎 fetch

这里有一篇文章讲述了[Ajax与Fetch的关系](https://github.com/camsong/blog/issues/2)

## 摘抄

XMLHttpRequest 是一个设计粗糙的 API，不符合**关注分离**（Separation of Concerns）的原则，配置和调用方式非常混乱，而且基于事件的异步模型写起来也没有现代的 Promise，generator/yield，async/await 友好。

> 在计算机科学中，关注点分离（SoC）是将计算机程序分解成function上尽可能less地重叠的不同特征的过程。 一个关心的问题是计划中的任何一个利益或重点。
> 个人目前的理解：将一个很复杂的事分成很多小事

使用 XHR 发送一个 json 请求一般是这样：

```js
var xhr = new XMLHttpRequest();
xhr.open('GET', url);
xhr.responseType = 'json';

xhr.onload = function() {
    console.log(xhr.response);
};

xhr.onerror = function() {
    console.log("Oops, error");
};

xhr.send();
```

使用 Fetch 后，顿时看起来好一点:（如果不用async与await其实也好不到哪去）

```js
fetch(url).then(function(response) {
    return response.json();
}).then(function(data) {
    console.log(data);
}).catch(function(e) {
    console.log("Oops, error");
});

```

## 正常人使用fetch的方式

[查看fetch具体的API可以点击这里](https://developer.mozilla.org/zh-CN/docs/Web/API/Fetch_API/Using_Fetch)

```js
class Panel extends Component {
    search = async () => {
        try {
                let response = await fetch(url);  // 尝试向服务器确认是否能进行通信
                let data = await response.json(); // 尝试向服务器获取数据
                console.log(data); // 这里拿到了数据
            } catch(e) {
                console.log("Oops, error", e);
        }
    }
}
```

## Fetch 常见坑

* Fetch 请求默认是不带 cookie 的，需要设置 `fetch(url, {credentials: 'include'})`
* 服务器返回 400，500 错误码时并不会 reject，只有网络错误这些导致请求不能完成时，fetch 才会被 reject。