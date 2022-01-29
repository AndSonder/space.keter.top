# 🍆 拓展

## [React.lazy](https://zh-hans.reactjs.org/docs/code-splitting.html#reactlazy)

`React.lazy` 函数能让你像渲染常规组件一样处理动态引入（的组件）。

使用之前：

```jsx
import OtherComponent from './OtherComponent';
```

使用之后：

```jsx
const OtherComponent = React.lazy(() => import('./OtherComponent'));
```

`React.lazy` 接受一个函数，这个函数需要动态调用 `import()`。它必须返回一个 `Promise`，该 Promise 需要 resolve 一个 `default export` 的 React 组件。

然后应在 `Suspense` 组件中渲染 lazy 组件，如此使得我们可以使用在等待加载 lazy 组件时做优雅降级（如 loading 指示器等）。

`fallback` 属性接受任何在组件加载过程中你想展示的 React 元素。你可以将 `Suspense` 组件置于懒加载组件之上的任何位置。你甚至可以用一个 `Suspense` 组件包裹多个懒加载组件。

```jsx{3-4,9-14}
import React, { Suspense } from 'react';

const OtherComponent = React.lazy(() => import('./OtherComponent'));
const AnotherComponent = React.lazy(() => import('./AnotherComponent'));

function MyComponent() {
  return (
    <div>
      <Suspense fallback={<div>Loading...</div>}>
        <section>
          <OtherComponent />
          <AnotherComponent />
        </section>
      </Suspense>
    </div>
  );
}
```

如果模块加载失败（如网络问题），它会触发一个错误。你可以通过[异常捕获边界（Error boundaries](/docs/react/extends#错误边界)）技术来处理这些情况。

```jsx{2,9,16}
import React, { Suspense } from 'react';
import MyErrorBoundary from './MyErrorBoundary';

const OtherComponent = React.lazy(() => import('./OtherComponent'));
const AnotherComponent = React.lazy(() => import('./AnotherComponent'));

const MyComponent = () => (
  <div>
    <MyErrorBoundary>
      <Suspense fallback={<div>Loading...</div>}>
        <section>
          <OtherComponent />
          <AnotherComponent />
        </section>
      </Suspense>
    </MyErrorBoundary>
  </div>
);
```

## [错误边界](https://zh-hans.reactjs.org/docs/error-boundaries.html)

部分 UI 的 JavaScript 错误不应该导致整个应用崩溃，为了解决这个问题，React 16 引入了一个新的概念 —— 错误边界。

错误边界是一种 React 组件，这种组件**可以捕获发生在其子组件树任何位置的 JavaScript 错误，并打印这些错误，同时展示降级 UI**，而并不会渲染那些发生崩溃的子组件树。

如果一个 class 组件中定义了 `static getDerivedStateFromError()` 或 `componentDidCatch()` 这两个生命周期方法中的任意一个（或两个）时，那么它就变成一个错误边界。

当抛出错误后，请使用 `static getDerivedStateFromError()` 渲染备用 UI ，使用 `componentDidCatch()` 打印错误信息。

```jsx{7-10,12-15,18-24}
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(error) {
        // 更新 state 使下一次渲染能够显示降级后的 UI
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        // 你同样可以将错误日志上报给服务器
        logErrorToMyService(error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            // 如果有错误
            return <h1>Something went wrong.</h1>;
        } else {
            // 如果没有错误，显示其子组件
            return this.props.children; 
        }
    }
}
```

然后你可以将它作为一个常规组件去使用，它的工作方式类似于 JavaScript 的 `catch {}`，不同的地方在于错误边界只针对 React 组件。

```jsx
<ErrorBoundary>
    <MyWidget />
</ErrorBoundary>
```
:::caution 注意
错误边界无法捕获以下场景中产生的错误：

* 事件处理（如果你需要在事件处理器内部捕获错误，使用普通的 JavaScript `try / catch` 语句）
* 异步代码（例如 `setTimeout` 或 `requestAnimationFrame` 回调函数）
* 服务端渲染
* 它自身抛出来的错误（并非它的子组件）
:::

## [Fragments](https://zh-hans.reactjs.org/docs/fragments.html)

它可以用来解决组件中过多的DOM节点，`key` 是唯一可以传递给 `Fragment` 的属性。未来 React 可能会添加对其他属性的支持，例如事件。

```jsx
function Hello(){
    return (
        <React.Fragment key={1}>
            <ChildA />
            <ChildB />
            <ChildC />
        </React.Fragment>
    );
}
```

还有一种新的短语法可用于声明它们。

```jsx
function Hello(){
    return (
        <>
            <ChildA />
            <ChildB />
            <ChildC />
        </>
    );
}
```

## [Context](https://zh-hans.reactjs.org/docs/context.html)

用于自顶向下传数据。下面这个例子中，`A`包裹`B`，`B`包裹`C`

```jsx{1,7-9}
const MyContext = React.createContext();

function A(){
    const [name, setName] = useState();

    return (
        <MyContext.Provider value={name}>
            <B/>{/*在这里的所有组件及其子组件都能获取value*/}
        </MyContext.Provider>
    )
}
```
* 类组件获取`context`的方式

```jsx{3,6}
class C extends Component {
    // 声明接受context
    static contextType = MyContext;

    render(){
        this.context //就能得到之前Provider的value值了。
    }
}
```

* 函数组件获取`context`的方式（类组件也能使用）

```jsx{3-5}
function C(){
    return (
        <MyContext.Consumer>
        {value => return <span>获取到的context{value}</span>}
        </MyContext.Consumer>
    )
}
```

其实使用hooks更加方便

```jsx
function C(){
    const value = useContext(MyContext);
    .....
}
```

## [Render Props](https://zh-hans.reactjs.org/docs/render-props.html)

术语 “render prop” 是指一种在 React 组件之间使用一个值为函数的 prop 共享代码的简单技术。

具有 render prop 的组件接受一个返回 React 元素的函数，并在组件内部通过调用此函数来实现自己的渲染逻辑。

```jsx
<DataProvider render={data => (
  <h1>Hello {data.target}</h1>
)}/>
```

```jsx
class DataProvider extends Component {
    render(
        return (<>
            {/* 这里可以预留非常多的参数会传到那个回调里 */}
            {/* 通过回调的返回值渲染组件出来 */}
            {this.props.render(a, b, c, d)}
        </>);
    );
}
```

## [PureComponent](https://zh-hans.reactjs.org/docs/react-api.html#reactpurecomponent)

`React.PureComponent` 与 `React.Component` 很相似。两者的区别在于 `React.Component` 并未实现 `shouldComponentUpdate()`，而 `React.PureComponent` 中以浅层对比 `prop` 和 `state` 的方式来实现了该函数。

如果赋予 React 组件相同的 `props` 和 `state`，`render()` 函数会渲染相同的内容，那么在某些情况下使用 `React.PureComponent` 可提高性能。

:::caution 注意
`React.PureComponent` 中的 `shouldComponentUpdate()` 仅作对象的浅层比较。如果对象中包含复杂的数据结构，则有可能因为**无法检查深层的差别，产生错误的比对结果**。仅在你的 `props` 和 `state` 较为简单时，**才使用** `React.PureComponent`，或者在深层数据结构发生变化时调用 forceUpdate() 来确保组件被正确地更新。你也可以考虑使用 immutable 对象加速嵌套数据的比较。

此外，`React.PureComponent` 中的 `shouldComponentUpdate()` **将跳过所有子组件树的 prop 更新。**因此，请确保所有子组件也都是“纯”的组件。
:::

## [Portals](https://zh-hans.reactjs.org/docs/portals.html)

它可以将需要渲染的组件，渲染在另一个组件下。（可以不是父组件）

第一个参数（`child`）是任何可渲染的 React 子元素，例如一个元素，字符串或 fragment。第二个参数（`container`）是一个 DOM 元素。

```jsx
render() {
    // React 并没有创建一个新的 div。它只是把子元素渲染到 `domNode` 中。
    // `domNode` 是一个可以在任何位置的有效 DOM 节点。
    return ReactDOM.createPortal(
        this.props.children, // 这是需要渲染的虚拟DOM
        domNode // 这个domNode是真实的dom，通过可以getElementById获取
    );
}
```

## [Profiler](https://zh-hans.reactjs.org/docs/profiler.html)

一个测量渲染性能的API。