# 动态规划
:::tip
1. dp常用的模型
2. 从不同的角度来介绍dp
:::
## DP方法论
![](https://gitee.com/coronapolvo/images/raw/master/20220204103614.png)

`DP问题的优化`
DP问题的优化一般都是对方程进行等价变形

`DP问题的思考路径`
拿到一个问题之后先考虑这个问题的状态表示，也就是这个问题可以被划分为多少个集合，每个集合的属性是什么。接下来去想如何进行状态计算，如何将当前集合进行划分才能够使得划分后的子集都能够计算出来。

## 背包问题
### 01背包问题
:::tip
有n个物品和v个背包，每个物品的体积为vi价值为w，`每件物品最多只能用一次`，调选一些物品让背包内的物品价值和最大
:::

#### 问题分析
首先我们先来分析一下状态表示，我们要如何去表示这个一个集合。
假设我们一共有n个物品和空间大小为v的背包，我们这个集合需要表示的含义就是当背包内只有i个物品的时候保证这i个物品的价值最大。集合的x轴表示背包中物品的个数，集合的y轴表示剩余的背包空间。集合中存储的是在有i个物品，空间还剩下j的时候背包内物品的最大价值。在朴素情况下我们使用二维数组`f[i][j]`来存储集合。

在考虑完了状态表示后我们继续考虑如何进行状态计算首先我们的集合可以划分为两个子集，一个子集是包含i一个子集是不包含i；对于不包含i的子集。对于不包含i的子集它的价值自然就是`f[i-1][j]`, 对于包含i的自己他的价值就是`f[i-1][j-vi] + wi` ，在进行更新的时候我们就取两者的最大值作为`f[i][j]`的数值。

#### 二维数组的写法
```c++
#include <iostream>
#include  <algorithm>

using namespace std;

const int N = 1010；

int n,m;
int v[N], w[N];
int f[N][N];

int main()
{
cin >> n >> m;
for(int i = 1;i <= n;i++) cin >> v[i] >> w[i];
for(int i = 1; i<= n; i++)
	for(int j=0;j<=m;j++)
	{
		f[i][j] = f[i-1][j];
		if(j >= v[i]) f[i][j] = max(f[i][j],f[i-1][j-v[i]] + w[i]);
	}
cout << f[n][m] << endl;
return 0;
}
```

#### 一维数组的写法
01背包问题可以进行优化，在进行二维处理的时候我们发现在使用计算第i行的时候只需要使用第i-1行的数值即可进行计算。也就是说可以使用一个滚动数组进行优化。需要注意的一个点就是由于在第二个for循环中如果j从0开始遍历，j-v[i]是在逐渐变大的就会导致使用已经覆盖过了的数组，所以第二重for循环需要倒着进行遍历；
```c++
#include <iostream>
#include  <algorithm>

using namespace std;

const int N = 1010；

int n,m;
int v[N], w[N];
int f[N];

int main()
{
cin >> n >> m;
for(int i = 1;i <= n;i++) cin >> v[i] >> w[i];
for(int i = 1; i<= n; i++)
	for(int j=m;j>=v[i];j--)
	{
		f[j] = max(f[j],f[j-v[i]] + w[i]);
	}
cout << f[m] << endl;
return 0;
}
```

### 完全背包问题
:::tip
每件物品可以用无限次
:::
#### 问题分析
我们还是从状态表示和状态计算两个方向进行分析；
和01背包类似，我们一开始也使用一个二维的数组来表示状态集合，这个集合中的元素表示所有只考虑前i个物品，且总体积不大于j的所有选法，集合的属性当然还是max啦。
01背包和完全背包问题最不一样的地方就在于如何进行状态计算，我们首先考虑如下如何进行集合的划分，现在由于物品可以使用无限次所以01背包的划分方案就不适用了，我们可以将问题进行拆解思考。假设现在背包的剩余空间是j，第i个物品的空间是vi，假设最多可以放k个i物品，那么情况就被分成了k+1种情况。这k+1种情况分别是：子集中有0个i，子集中有1个i，... 子集中有k个i，我们在更新集合的时候需要选取全部子集中的最大值即可。对于子集中有x个i的情况下物品的价值就是`f[i-1][j-k*v[i] + k * w[i]`

#### 朴素写法
```c++
#include <iostream>
using namespace std;

const int N = 1010;
int v[N],w[N];
int f[N][N];

int main()
{
    int n,m;
	cin >> n >> m;
	for(int i=1;i<=n;i++) cin >> v[i] >> w[i];
	for(int i=1;i<=n;i++)
		for(int j=0;j<=m;j++)
			for(int k=0;k*v[i] <= j;k++)
				f[i][j] = max(f[i][j],f[i-1][j-k*v[i]] + k*w[i]);
	cout << f[n][m];
}
```

#### 优化
我们可以看到现在的朴素算法需要使用三重for循环，速度很忙，我们可以再进行一波优化。
先来观察现在的公式：
`f[i][j] = max(f[i-1][j],f[i-1][j-v]+w,f[i-1][j-2v]+2w,f[i-1][j-3v]+3w)`
接下来我们来对比一下`f[i][j-v]`：
`f[i][j-v] = max(f[i-1][j-v],f[i-1][j-2v]+w,f[i-1][j-3v]+2w)`
然后我们就可以发现：
`f[i][j] = max(f[i-1][j],f[i][j-v]+w)`
同样的我们也可以用滚动数组来进行优化：
`f[j] = max(f[j],f[j-v]+w)`
而且这里还没有01背包的问题，j是从小到大枚举的，j-v一定是小于j的所以都是计算过的没有什么大问题。

```c++
#include <iostream>
using namespace std;

const int N = 1010;

int n,m;
int v[N],w[N];
int f[N];

int main()
{
	cin >> n >> m;
	for(int i=1;i<=n;i++) cin >> v[i] >> w[i];
	for(int i=1;i<=n;i++)
		for(int j=v[i];j<=m;j++)
			f[j] = max(f[j],f[j-v[i]]+w[i]);
	cout << f[m];
}
```

### 多重背包问题
:::tip
每件物品的数量不一样
:::
多重背包问题的朴素问题和完全背包问题基本一模一样
#### 朴素写法
```c++
#include <iostream>
using namespace std;
const int N = 110;

int v[N],w[N],s[N];
int f[N][N];

int main()
{
	int n,m;
	cin >> n >> m;
	for(int i=1;i<=n;i++) cin >> v[i] >> w[i] >> s[i];
	for(int i=1;i<=n;i++)
		for(int j=0;j<=m;j++)
			for(int k=0;k<=s[i] and k*v[i] <= j;k++) 
				f[i][j] = max(f[i][j], f[i-1][j - v[i]*k] + w[i]*k);
	cout << f[n][m];
}
```
#### 优化
这种优化的方案就非常牛皮了，这个优化方案的思路是将多重背包问题看做01背包问题来解决。
我们现在假设有4个物品i，我们将物品i进行分组，分组的目的是`让分组之后的组别可以任意组合成1-4之间的任意个数的物品i`。这里有一个已经存在的定理就是`如果想让分组之后的可以组合成1-n中的任何数，只需要按照2的倍数进行分类即可`。
再举一个更加具体的例子，假如有5个物体i，那么我们就将物体i分为 [1个i, 2个i, 4个i], 这个时候我们将1个i看做一个新的物体，2个i也看做一个新的物体，这样进行分类之后我们就可以将不同数量的i都看做不同的新的物体。然后就可以进一步转换为一个01背包问题了。
```c++
#include <iostream>
using namespace std;

const int N = 25000,M = 2010;

int n,m;
int v[N],w[N];
int f[N];

int main()
{
	cin >> n >>m;
	int cnt = 0;
	for(int i = 1;i<=n;i++)
	{
		int a, b ,s;
		cin >> a >> b >> s;
		int k = 1;
		while(k <= s)
		{
			cnt ++;
			v[cnt] = a * k;
			w[cnt] = b * k;
			s -= k;
			k *= 2;
		}
		if(s > 0)
		{
			cnt ++;
			v[cnt] = a * s;
			w[cnt] = b * s;
		}
	}
	n = cnt;
	for(int i=1;i<=n;i++)
		for(int j=m;j>=v[i];j--)
			f[j] =  max(f[j],f[j-v[i]] + w[i]);
	cout <<  f[m];
}
```


### 分组背包问题
:::tip
物品有若干组，每一组物品是互斥的
:::
这个问题没有什么要去优化的地方，主要就是状态计算那你注意一下就可以，选择所有情况下空间最大的情况就可以了
```c++
#include <iostream>
using namespace std;

const int N = 110;
int v[N][N],w[N][N],s[N];
int f[N];

int main()
{
	int n,m;
	cin >> 	n >> m;
	for(int i=1;i<=n;i++)
	{
		cin >> s[i];
		for(int j=0;j<s[i];j++) cin >> v[i][j] >> w[i][j];
	}
	for(int i=1;i<=n;i++)
	    for(int j=m;j>=0;j--)
	        for(int k=0;k<s[i];k++)
	            if(v[i][k]<=j)
	                f[j] = max(f[j],f[j-v[i][k]] + w[i][k]);
    cout << f[m];
}
```
