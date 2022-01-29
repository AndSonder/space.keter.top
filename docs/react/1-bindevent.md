# 🍿 React事件绑定

## 原生事件绑定

它会把引号里的东西当成JavaScript去执行。

```html
<button onclick="nativeClick()">按钮</button>
```

```js
function nativeClick(){
    // todo handle click
}
```

## React事件绑定

:::caution
这样写的this指向有问题
:::

```jsx{8-11,17-20}
class Weather extends React.Component{
    constructor(props){
        super(props);
        this.state = {isHot: true};
    }

    render(){
        // 注意这里的驼峰
        // 传给onClick的是一个函数！！！必须是一个函数
        // 这个函数在事件被触发时，React自动帮你调用
        return <h1 onClick={this.handleOnClick}>
            今天天气很{this.state.isHot ? "炎热" : "凉爽"}
        </h1>;
    }

    handleOnClick() {
        // 这样写，无法取到真正的this
        // 这个函数被调用时，是React帮你调用的
        // 不是Weather实例对象调用的，weather.handleOnClick()
        // 类中的方法默认开启局部的严格模式，所以这里的this为undefined
    }
}
```

## 解决类实例方法this的指向

:::tip 方法一
通过 `Function.prototype.bind()` 方法

`bind`能做到两件事，返回一个新的函数，修改这个函数的`this`
:::

```jsx{6-7}
class Weather extends React.Component{
    constructor(props){
        super(props);
        this.state = {isHot: true};

        // 拿着原型上的handleOnClick，生成了一个新的handleOnClick，挂到了实例化的组件上
        this.handleOnClick = this.handleOnClick.bind(this);
    }

    render(){
        return <h1 onClick={this.handleOnClick}>
            今天天气很{this.state.isHot ? "炎热" : "凉爽"}
        </h1>;
    }

    handleOnClick() {
        // 这里可以拿到this
    }
}
```

:::tip 方法二
使用箭头函数，这里顺便把state给一起简写了
:::

```jsx{3,11-14}
class Weather extends React.Component{

    state = {isHot: true, wind: "微风"};

    render(){
        return <h1 onClick={this.changeWeather}>
            今天天气很{this.state.isHot ? "炎热" : "凉爽"}
        </h1>;
    }

    changeWeather = () => {
        const isHot = this.state.isHot;
        this.setState({isHot: !isHot});
    }
}
```