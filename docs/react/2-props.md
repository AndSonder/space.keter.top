# 🍊 Props 

一般用父组件的`state`，给予给子组件的属性，就是`props`，而且`props`是只读的。

## 基本使用 

```jsx{3,7-8,15}
class Person extends React.Component{
    render(){
        const {name, age} = this.props;

        return (
            <ul>
                <li>姓名：{name}</li>
                <li>年龄：{age}</li>
            </ul>
        );   
    }
}

ReactDOM.render(
    <Person name="Therainisme" age={19}/>, 
    document.getElementById("root")
);
```

## 批量传递props 

```jsx{14,17}
class Person extends React.Component{
    render(){
        const {name, age} = this.props;

        return (
            <ul>
                <li>姓名：{name}</li>
                <li>年龄：{age}</li>
            </ul>
        );   
    }
}

const pdate = {name: "Therainisme", age: 19};

ReactDOM.render(
    <Person {...pdate}/>, 
    document.getElementById("root")
);
```

:::tip 补充知识
展开运算符能配合字面量声明创建一个复制对象，但是它不能打印出来。

所以上面的 `<Person {...pdate}/>` 能这么写完全是babel的功劳
:::

```js
let A = {name: "Therainisme", age: 19};
let B = {...A}; // 展开运算符能这样复制一个对象

console.log(...A); // 报错，展开运算符不能展开对象
console.log(B); // 复制的对象
```

## 对props进行限制

:::tip 注意
在React15之前，还能`React.PropTypes`去赋值，但是之后React开发人员怕React包太大，于是就把它就独立出来了。

嗯，需要引入 prop-types库
:::

```jsx{14-23}
class Person extends React.Component{
    render(){
        const {name, age} = this.props;

        return (
            <ul>
                <li>姓名：{name}</li>
                <li>年龄：{age}</li>
            </ul>
        );   
    }
}

// 如果给一个类加上这个对象，React就会对其进行限制
Person.propTypes = {
    name: PropTypes.string.isRequired, // 字符串类型 必填
    age: PropTypes.number, // 数字类型
    speak: PropTypes.func  // 函数，不是function因为function是关键字
}

Person.defaultProps = {
    age: "18" // 默认永远18岁
}

const pdate = {name: "Therainisme", age: 19};

ReactDOM.render(
    <Person {...pdate}/>, 
    document.getElementById("root")
);
```

## props的简写

:::tip 需要知道的知识
给一个类上加属性 `Person.A = 1`，等于在类体中使用`static`关键字。
:::

```jsx{13-21}
class Person extends React.Component{
    render(){
        const {name, age} = this.props;

        return (
            <ul>
                <li>姓名：{name}</li>
                <li>年龄：{age}</li>
            </ul>
        );   
    }

    static propTypes = {
        name: PropTypes.string.isRequired,
        age: PropTypes.number,
        speak: PropTypes.func
    }

    static defaultProps = {
        age: "18"
    }
}

const pdate = {name: "Therainisme", age: 19};

ReactDOM.render(
    <Person {...pdate}/>, 
    document.getElementById("root")
);
```

## contructor与props

[官网](https://zh-hans.reactjs.org/docs/react-component.html?#constructor)在这里写得非常清楚。**如果不初始化 state 或不进行方法绑定，则不需要为 React 组件实现构造函数。**

在实际的开发过程中很少写构造器

```jsx{2-6}
class Person extends React.Component{
    constructor(props) {
        super(props);
        // 如果想要在这里访问 this.props，必须调super
        // 如果在这里不这样调用 this.props，那么构造器的super和props随便写
    }
    ...
}
```

## 在函数是组件中使用props

如果不使用hooks，函数试组件只能使用props......

```jsx
function Person(props){
    const {name, age} = this.props;

    return (
        <ul>
            <li>姓名：{name}</li>
            <li>年龄：{age}</li>
        </ul>
    );   
}

Person.propTypes = {
    name: PropTypes.string.isRequired,
    age: PropTypes.number, 
    speak: PropTypes.func 
}

Person.defaultProps = {
    age: "18"
}
```

## 最后需要理解的

* props的作用是通过标签属性，从组件外向组件内传递变化的数据
* 而且组件内部无法修改props数据，它是只读的