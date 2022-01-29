# 🌺 Hooks

## [useState](https://zh-hans.reactjs.org/docs/hooks-reference.html#usestate)

```jsx
const [state, setState] = useState(initialState);
```

一般state就是一个值，不是一个对象。完全取决你怎么写。

:::caution 注意

与 class 组件中的 `setState` `方法不同，useState` 不会自动合并更新对象。你可以用函数式的 `setState` 结合展开运算符来达到合并更新对象的效果。

:::

```jsx
const [state, setState] = useState({});
setState(prevState => {
  // 也可以使用 Object.assign
  return {...prevState, ...updatedValues};
});
```

[useReducer](/docs/react/hooks#usereducer) 是另一种可选方案，它更适合用于管理包含多个子值的 `state` 对象。

## [useEffect](https://zh-hans.reactjs.org/docs/hooks-reference.html#useeffect)

```jsx
useEffect(() => {
    /*
    这部分是创建时执行的代码
    */
    return () => {
        // 销毁时执行的回调
    };
}, [constraint, ....]); // 约束变量，当这个变量改变时，这个Effect便会销毁，再创建新的
```

## [useRef](https://zh-hans.reactjs.org/docs/hooks-reference.html#useref)

```jsx
const refContainer = useRef(initialValue);
```

`useRef` 返回一个可变的 ref 对象，其 `.current` 属性被初始化为传入的参数`initialValue`。返回的 ref 对象在组件的整个生命周期内持续存在。

```jsx{2,5,9}
function TextInputWithFocusButton() {
    const inputEl = useRef(null);
    const onButtonClick = () => {
        // `current` 指向已挂载到 DOM 上的文本输入元素
        inputEl.current.focus();
    };
    return (
    <>
        <input ref={inputEl} type="text" />
        <button onClick={onButtonClick}>Focus the input</button>
    </>
    );
}
```

## [useContext](https://zh-hans.reactjs.org/docs/hooks-reference.html#usecontext)

可以获取上层组件通过[Context](/docs/react/extends#context)传来的`value`

```jsx
const value = useContext(MyContext);
```

## [useReducer](https://zh-hans.reactjs.org/docs/hooks-reference.html#usereducer)

```jsx
const [state, dispatch] = useReducer(reducer, initialArg, init);
```

* `state`：被管理的状态
* `dispatch`：一个函数，和 redux 的一样
* `reducer`：一个函数，和 redux 的一样
* `initialArg`：`state` 的初始值
* `init`：一个函数，返回值是 `state` 的初始值（一般不用这个）

:::caution 注意
React 不使用 `state = initialState` 这一由 Redux 推广开来的参数约定。有时候初始值依赖于 props，因此需要在调用 Hook 时指定。如果你特别喜欢上述的参数约定，可以通过调用 `useReducer(reducer, undefined, reducer)` 来模拟 Redux 的行为，但 React 不鼓励你这么做。
:::

## [useCallback](https://zh-hans.reactjs.org/docs/hooks-reference.html#usecallback)

把内联回调函数及依赖项数组作为参数传入 `useCallback`，它将返回该回调函数的 memoized 版本，该回调函数仅在某个依赖项改变时才会更新。

```jsx
const memoizedCallback = useCallback(() => {
    doSomething(a, b);
}, [a, b],);
```

效果如下：（请注意闭包的问题！）

```js
// 这样调用
memoizedCallback();

// 和这样调用是一样的
() => {
    doSomething(a, b);
}
```

## [useMemo](https://zh-hans.reactjs.org/docs/hooks-reference.html#usememo)

```jsx
const memoizedValue = useMemo(() => computeExpensiveValue(a, b), [a, b]);
```

`memoizedValue`的值是回调函数的返回值，这个值只有当`[a, b]`其中一个改变时，再次执行回调函数才会变更。

记住，传入 `useMemo` 的函数会在渲染期间执行。请不要在这个函数内部执行与渲染无关的操作，诸如副作用这类的操作属于 `useEffect` 的适用范畴，而不是 `useMemo`。

你可以把 `useMemo` 作为性能优化的手段，但不要把它当成语义上的保证。

## [useImperativeHandle](https://zh-hans.reactjs.org/docs/hooks-reference.html#useimperativehandle)

```jsx
useImperativeHandle(ref, createHandle, [deps]);
```

`useImperativeHandle` 应当与 `forwardRef` 一起使用。`useImperativeHandle` 可以让你在使用 ref 时自定义暴露给父组件的实例值。在大多数情况下，应当避免使用 ref 这样的命令式代码。

```jsx{3-7,10}
function FancyInput(props, ref) {
    const inputRef = useRef();
    useImperativeHandle(ref, () => ({
        focus: () => {
            inputRef.current.focus();
        }
    }));
    return <input ref={inputRef}/>;
}
FancyInput = forwardRef(FancyInput);
```

在本例中，渲染 `<FancyInput ref={inputRef} />` 的父组件可以调用 `inputRef.current.focus()`。

## [useLayoutEffect](https://zh-hans.reactjs.org/docs/hooks-reference.html#uselayouteffect)

其函数签名与 useEffect 相同。

这个是用在处理DOM的时候,当你的 `useEffect` 里面的操作需要处理DOM,并且会改变页面的样式,就需要用这个,否则可能会出现出现闪屏问题, `useLayoutEffect` 里面的 `callback` 函数会在DOM更新完成后立即执行,但是会在浏览器进行任何绘制之前运行完成,阻塞了浏览器的绘制.

这里有一篇博客讲得非常好：[useEffect和useLayoutEffect的区别](https://www.jianshu.com/p/412c874c5add)。

## [useDebugValue](https://zh-hans.reactjs.org/docs/hooks-reference.html#usedebugvalue)

```jsx
useDebugValue(value)
```

`useDebugValue` 可用于在 React 开发者工具中显示自定义 hook 的标签。

```jsx
function useFriendStatus(friendID) {
    const [isOnline, setIsOnline] = useState(null);

    // 在开发者工具中的这个 Hook 旁边显示标签
    // e.g. "FriendStatus: Online"
    useDebugValue(isOnline ? 'Online' : 'Offline');

    return isOnline;
}
```