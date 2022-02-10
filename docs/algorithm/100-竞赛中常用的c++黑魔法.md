# 竞赛中常用的c++黑魔法


## C++中的STL
- vector：变长数组，倍增的思想
- pair：存储一对数
- string：字符串，substr(), c_str()
- queue：push(), front(), pop()
- priority_queue：优先队列，push(), top(), pop()
- stack：栈，push(), top(), pop()
- deque：双端队列
- set, map, multiset, multimap：基于红黑树来实现，本质上是动态维护一个有序序列
- unordered_set, unordered_map, unordered_multiset, unordered_multimap：哈希表
- bitset：压位

### vector
```c++
vector<int> a(10,3); // 定义一个长度为10的vector，初始化为3；
a.size(); // vector的size，所有容器都有
a.empty(); // 范围vector是否为空，所有容器都有
a.clear(); // 清空
a.front(); // 第一个数
a.back(); // 最后一个数
a.push_back(); // 在最后插入一个数
a.pop_back(); // 删除最后一个数
// vector支持比较运算
vector<int> a(4,3),b(3,4);
if(a > b) cout << "Yes";
else cout << "No"
```

### pair
```c++
pair<int,int> a;
a = {20,"abc"};
a.first(); // 获取第一个元素
a.second(); // 获取第二个元素
// pair也能进行sort
```

### string
```c++
string a = "Acwing";
a.size(); // 获取string的大小
a.empty(); // 判断是否为空
a.clear(); // 清空
a += "def";
cout << a. substr(1,2) << endl; // 第一个参数起始位置，第二个参数是字符串长度
```

### query
```c++
query<int> a;
a.size();
a.empty();
a.push(1); // 队尾插入元素
a.front(); // 返回队头元素
a.back(); // 返回队尾元素
a.pop(); // 删除队头元素
```

### priority_queue
```c++
// 默认是大根堆
priority_queue<int> heap;
heap.clear();
heap.size();
heap.empty();
// 如何定义一个小根堆： 1. 插入负数 2. 直接定义
heap.push(-x); // 黑科技方法
priority_queue<int,vector<int>,greater<int>> q;
```
### stack
```c++
stack<int> s;
s.size();
s.empty();
s.push();
s.top();
s.pop();
```
### deque
```c++
deque<int> a;
a.size();
a.empty();
a.clear();
a.front();
a.back();
a.push_back();
a.pop_back();
```

### set/multiset
```c++
set<int> s; // 不能有重复元素
// s.begin()/end()
multiset<int> MS; // 可以有重复元素
s.insert(1); 插入一个数
s.size();
s.empty();
s.clear();
	s.find(1); // 查找一个元素，如果不存在的话返回end迭代器
s.erase(1); // 输入是一个数x，输出所有x （2）输入一个迭代器，删除这个迭代器
// set 最核心的操作
s.lower_bound(); // 范围大于等于x的最小的数
s.upper_bound(); // 返回大于x的最小的数
```
### map/multimap
```c++
#include <map>
// 和python里面的字典非常的相似
map<string,int> a;
a["2"] = 3;
a.insert({"1",1});
a.erase({"1",1});
a.find({"1",1});
```

> unordered_set, unordered_map, unordered_multiset, unordered_multimap的操作和set或者map等的操作基本一致，唯一的区别就是不支持类似lower_bound()这样的操作 （哈希表的内部是无序的）

### biset
> 可以省下来8位的空间

```c++
bitset<10000> s;
// 支持所有的基本操作：
// 移位操作：<< >> 
// == !=
// count() 返回有多少个1
// any() 判断是否至少有一个1
// none() 判断是否全为0
// set()，把所有为置为1
// set(k,v), 将第k个变为v
// reset(), 把所有位变成0
// flip(), 把所有位取反
```


