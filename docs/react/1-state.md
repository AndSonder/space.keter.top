# 🍉 State 

state是组件对象最重要的属性

## 声明方法 

```js{3-5}
class Weather extends React.Component{
    constructor(props){
        // 如果B继承了A，且B中写了构造器，那么A类构造器中的super是必须要调用的
        // 这是js构造器规定的，不是React规定的
        super(props);

        // React官方要求写成一个对象
        // 如果没有这一步，在实例化的组件中是 null，在早一点的React版本甚至是 {}
        this.state = {isHot: true};
    }

    render(){
        // 在构造器中已经将state写入到this中
        // 这里就可以直接用了
        return <h1>今天天气很{this.state.isHot ? "炎热" : "凉爽"}</h1>;
    }
}
```

## setState 

:::tip 各函数调用次数
`constructor`： 1次

`render`：1 + n次

第一次渲染的时候，`constructor` => `render`

在后续的事件触发后，调用`setState`的时候，`render`会帮你刷新页面
:::

```jsx{4,18-22}
class Weather extends React.Component{
    constructor(props){
        super(props);
        this.state = {isHot: true, wind: "微风"};

        this.changeWeather = this.changeWeather.bind(this);
    }

    render(){
        return <h1 onClick={this.changeWeather}>
            今天天气很{this.state.isHot ? "炎热" : "凉爽"}
        </h1>;
    }

    changeWeather() {
        const isHot = this.state.isHot;

        // 如果直接更改，页面不会刷新
        // this.state.isHot = false; 不可取

        // state必须通过setState更新，且更新是一种合并，不是替换
        this.setState({isHot: !isHot});
    }
}
```

## state的简写方式 

这里顺便把函数绑定`this`的方式给一起简写了，直接把`constructor`给省略

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

## setState更新状态的2种写法 

1. `setState(newState, [callback])`
* `newState`：新的`state`对象
* `callback`：可选的回调函数，在状态更新完毕后会调用

2. `setState(updater(state, props), [callback])`
* `updater`：它的返回值为新的`state`对象，可以接受`state`和`props`。
* `callback`：可选的回调函数，在状态更新完毕后会调用

## 一些需要注意的 

* 组件中的`render`方法中的`this`为组件实例对象
* 组件自定义方法中的`this`为`undefined`，有两种解决方式
  1. `Function.prototype.bind()`
  2. `() => {}`
* 不能直接修改，需要用`setState()`，这样`render`才会二次调用
