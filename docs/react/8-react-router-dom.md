# 🍑 React路由

先把它的官网直接丢上来 [reactrouter](https://reactrouter.com/)，Web端它的相关包是 `react-router-dom`

## BrowserRouter

路由最重要的一个组件，用于注册路由。它必须包裹所有 react-router-dom 所属的路由组件，这些组件才具有对应的路由属性。它一般是写在脚手架的最外端，也就是 `index.jsx` 中。

```jsx
ReactDOM.render(
    <BrowserRouter>
        <App/>
    </BrowserRouter>,
    document.getElementById('root')
);
```

## HashRouter

与`BrowserRouter`相似，只不过它是以锚点为驱动。

1. 底层原理不一样
* `BrowserRouter`使用的是H5的history API，不兼容IE9以下版本
* `HashRouter`使用的是URL的哈希值

2. url表现形式不一样
* `BrowserRouter`的路径中没有`#`，`localhost:3000/welcome`
* `HashRouter`的路径包含`#`，`localhost:3000/#welcome`

3. 刷新后对路由state参数的影响
* `BrowserRouter`的保存在`history`对象中，不会丢
* `HashRouter`会丢

## Route

该组件监听浏览器的地址栏，不同的地址栏会匹配不同的组件，用于展示区。路由匹配并渲染时，会给`component`传入的组件添加 `history`、`location`和`match`属性（props）。

```jsx
<Route path="/about" component={About}/>
```

## Link

`<Link/>`组件用于替代传统的`<a/>`，且可以在页面跳转时传递参数，用于导航区。

### 基本用法

`push`模式跳转

```jsx
<Link to="/about"></Link>
```

也可以`replace`模式跳转

```jsx
<Link replace to="/about"></Link>
```

### 传递params参数

一个声明（`Route`）接收，一个负责（`Link`）传。

```jsx
<Route path="/welcome/:id" component={Welcome}/>
......
<Link to="/welcome/520"></Link>
```

传递之后的参数在 `this.props.match.params` 里。

```jsx{3-4}
class Welcome extends Component {
    render(){
        // params是一个对象，{id: 520}
        const params = this.props.match.params;
    }
}
```

### 传递search参数

它就像`query`参数，在浏览器地址栏`?`后面的数据（urlencoded编码）。`Route`组件无需声明接收。

```jsx
<Link to="/welcome?id=520"></Link>
```

传递之后的参数在 `this.props.location.search` 里。

```jsx{5-8}
import qs from 'querystring'; // React脚手架已经帮忙下载好了

class Welcome extends Component {
    render(){
        // search是一个字符串 "?id=520"
        const search = this.props.location.search;
        const result = qs.parse(search.slice(1)); // 去除'?'号
        // result = {id: 520}；
    }
}
```

### 传递state参数

把`Link`组件的`to`参数稍微改动一下，增加一个`state`参数，同样的`Route`组件无需声明接收。刷新也可以保留住参数。

```jsx
<Link to={{pathname: "/welcome", state: {id: 520}}}></Link>
```

传递之后的参数在 `this.props.location.state` 里。

```jsx{3-7}
class Welcome extends Component {
    render(){
        // state是一个对象 {id: 520}
        // || {}
        // 如果是HashRouter，history可能会丢，所以给它一个默认对象
        // 如果是BrowserRouter则不受影响
        const state = this.props.location.state || {}; 
    }
}
```

## NavLink

比与`<Link/>`组件多一个功能。点击时会给类名加上 `active` ，而且 `<NavLink/>` 会保证只有一个导航按钮的类名拥有 `active`。

当然如果使用的组件库不是 `active`，也可也认为的控制加的是其他的类名。

```jsx
<NavLink activeClassName="highlight" children="跳转"/>
```

## Switch

用于多个路由都匹配时，只展示最先被匹配的那个。如果不使用它，类似的效果就像 `switch` 语句的 `case` 不加 `break`。

```jsx
<Switch>
    <Route path="/introduction" component={Article} />
    <Route path="/" component={Welcome} />
</Switch>
```

## 路由匹配模式

默认的匹配方式是模糊匹配

```jsx
<Link to="/home"/>    // 能
<Link to="/home/b"/>  // 能
<Link to="a/home/b"/> // 否
......
<Route path="/home"/>
```

若想严格匹配，给`<Route/>`组件加一个`exact`。

```jsx{5}
<Link to="/home"/>    // 能
<Link to="/home/b"/>  // 能
<Link to="a/home/b"/> // 否
......
<Route exact path="/home"/>
```

## Redirect

重定向组件，可以用在 `Switch`里，当所有路由都不匹配时，可以用作一个兜底的路由。

```jsx{4}
<Switch>
    <Route path="/introduction" component={Article} />
    <Route path="/welcome" component={Welcome} />
    <Redirect to="/error" component={Error}/>
</Switch>
```

## 编程式路由导航

借助`props`中的`history`进行。

### push跳转

```js
this.props.history.push(`/welcome/520`);          // params        
this.props.history.push(`/welcome?id=520`);       // search
this.props.history.push(`/welcome`, {id: 520});   // state
```

### replace跳转

```js
this.props.history.replace(`/welcome/520`);        // params
this.props.history.replace(`/welcome?id=520`);     // search
this.props.history.replace(`/welcome`, {id: 520}); // state
```

### go跳转

```js
this.props.history.go(step);    // step是前进还是后退number类型的步数 
this.props.history.goForward(); // 前进
this.props.history.goBack();    // 后退
```

## withRouter

用于普通组件赋予路由组件的属性（`history`、`location`和`match`）

```jsx{5}
class Header extends Component {
    ......
}

export default withRouter(Header);
```