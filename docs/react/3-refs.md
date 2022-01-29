# 🍋 Refs 

Refs一般用于获取元素的DOM对象，用于替代JavaScript的操作DOM的方法。

## 字符串形式Refs 

:::caution
React现在已经不推荐写字符串形式的Refs了，可能在未来的版本会被移除
:::

```jsx{3-6,9}
class Demo extends React.Component {

    btn = () => {
        // 这样能拿到按钮的DOM对象
        console.log(this.refs.btnElement);
    }

    render(){
        <button ref="btnElement" onClick={this.btn}></button>
    }
}
```

## 回调Refs

```jsx{4,8-10}
class Demo extends React.Component {

    btn = () => {
        console.log(this.btnElement);
    }

    render(){
        // ref里写回调，在组件渲染时会把DOM对象传入回调函数
        // this.btnElement = e 就是把DOM对象往Demo组件实例对象上挂一个
        <button ref={e => this.btnElement = e} onClick={this.btn}></button>
    }
}
```

:::danger 关于回调refs的说明 
如果 `ref` 回调函数是以**内联函数**的方式定义的，在更新过程中它会被执行两次，**第一次传入参数 `null`，然后第二次会传入参数 DOM 元素。**

这是因为在每次渲染时会创建一个新的函数实例，所以 React 清空旧的 ref 并且设置新的。通过将 ref 的回调函数定义成 class 的绑定函数的方式可以避免上述问题，**但是大多数情况下它是无关紧要的**。

[来源React官网](https://zh-hans.reactjs.org/docs/refs-and-the-dom.html#caveats-with-callback-refs)
:::

下面这个例子可以解决内联函数被调用两次的问题。下面例子的ref每次识别到回调函数都是同一个。如果是内联函数，每次ref回调调用都会创建一个新的回调函数。React会帮你做一个初始化的动作，先传`null`，再传对应的DOM对象。

```jsx{7-9,12}
class Demo extends React.Component {

    btn = () => {
        console.log(this.btnElement);
    }

    saveBtnElement = e => {
        this.btnElement = e;
    }

    render(){
        <button ref={this.saveBtnElement} onClick={this.btn}></button>
    }
}
```

## createRef

```jsx{3-5,8-9,13}
class Demo extends React.Component {

    // React.createRef调用后可以返回一个容器
    // 该容器可以储存被ref所标识的节点
    btnRef = React.createRef();

    btn = () => {
        // 节点DOM在ref容器的current上
        console.log(this.btnRef.current);
    }

    render(){
        <button ref={this.btnRef} onClick={this.btn}></button>
    }
}
```