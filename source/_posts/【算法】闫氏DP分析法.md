---
title: 「算法」闫氏DP分析法
date: 2021-05-30 14:01:02
tags: 算法
categories: 算法
katex: true
---



# 闫氏DP分析法

- 核心：从集合角度来分析DP问题；
- 目的：求有限集中的最值

## 动态规划

### 状态表示 f(i)

- 化0为整，把一类集合变成一个整体然后用一个数来表示它；

1. 集合
2. 属性：f(i)与集合的关系 （max/min/bool）

### 状态计算

化整为0的过程：将f(i) 划分为几个子集进行计算，分别求每个子集，最后将子集合并起来；

- 划分的依据：寻找最后一个不同点

# 几个例题

## 01背包问题

<img src="https://gitee.com/coronapolvo/images/raw/master/2021053017091320210530154153image-20210530154147989.png" alt="image-20210530154147989" style="zoom:50%;" />

选择问题

`状态表示:` f(i,j)

1. 集合：所有只考虑前i个物品，并且总体积不超过j的选法的集合；
2. 属性：max 集合当中每一个方案的最大价值

<img src="https://www.zhihu.com/equation?tex=f(n,v)" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">就是只考虑前n个值，不超过v的最大的值

`状态计算:`

对于f(v,i), 可以分为两类，一类是不选择第i个物品的方案，一类是选择第i个物品的方案； 

那么我们可以得到不选择i的物品的方案就是f(v,i-1), 选择第i个物品的最大值是<img src="https://www.zhihu.com/equation?tex=f(i-1,j-v_i)+w_i" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">

那么最后的结果就是：<img src="https://www.zhihu.com/equation?tex=max(f(v,i-1),f(i-1,j-v_i)+w_i)" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">

AC代码为：

```c++
#include<iostream>

using namespace std;
const int N = 2010;

int n,m;
int v[N],w[N];
int f[N][N];

int main(){
	cin >> n >> m;
  for (int i=1;i<=n; i++) cin >> v[i] >> w[i];
  for(int i = 1; i <= n; i++){
    for(int j = 0; j <= m; j++){
      f[i][j] = f[i-1][j];
      if(j >= v[i]) f[i][j] = max(f[i][j],f[i-1][j-v[i]] + w[i]);
    }
  }
  cout << f[n][m];
}
```

## 完全背包问题

![image-20210530180439882](https://gitee.com/coronapolvo/images/raw/master/20210530180441image-20210530180439882.png)

`状态表示:` f(i,j)

1. 集合：所有只考虑前i个物品，并且总体积不超过j的选法的集合；
2. 属性：max 集合当中每一个方案的最大价值

<img src="https://www.zhihu.com/equation?tex=f(n,v)" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">就是只考虑前n个值，不超过v的最大的值

`状态计算:`

对于f(i,j), 完全背包问题需要划分为多个集合；

![image-20210530180957004](https://gitee.com/coronapolvo/images/raw/master/20210530181001image-20210530180957004.png)

```python
/*
01背包问题：f(i,j) = max(f[i-1][j], f[i-1]f[j-v] + w)
完全背包问题：f(i,j) = max(f[i-1][j], f[i]f[j-v] + w)
*/

#include <iostream>
using namespace std;
const int N= 2010;
int n,m;
int v[N],w[N];
int f[N][N];
int main(){
    cin >> n >>m;
    for (int i=1;i<=n;i++) cin >> v[i] >> w[i];
    
    for (int i=1;i<=n;i ++){
        for(int j=0;j<=m;j++){
            f[i][j] = f[i-1][j];
            if (j >= v[i]) f[i][j] = max(f[i][j], f[i][j-v[i]]+w[i]);
        }
        
        }
    cout << f[n][m] << endl;
}
```

当然你还可以进行优化：

```c++
#include <iostream>
using namespace std;
const int N = 1010;

int n,m;
int v[N],w[N];
int f[N]
int main(){
  cin >> n >> m;
  for(int i=1; i<=m; i++) cin >> v[i] >> w[i];
  for(int i=1; i<=n; i++){
    for(int j=v[i];j<=m,j++){
      f[j] = max(f[j],f[j-v[i]] + w);
    }
  }
  return 0;
}
```

## 石子问题

题目： [https://www.acwing.com/problem/content/description/284/](https://www.acwing.com/problem/content/description/284/)

![image-20210530163115020](https://gitee.com/coronapolvo/images/raw/master/20210530170927image-20210530163115020.png)

`状态表示:` f(i,j)

1. 集合：所有将i到j的区间合并成一堆的方案的集合；
2. 属性：min 集合当中每一个方案的最小代价；

`状态计算:`

对于f(i,j) 需要分为多个集合；

最后我们需要计算的结果就是 f(1,j)

```c++
#include <iostream>

using namespace std;

const int N = 310;
int s[N];
int f[N][N];
int n;
int main(){
  cin >> n;
  for (int i=1;i<=n;i++) cin >> s[i],s[i]+=s[i-1];
  for(int len =2 ; len <= n; len ++){
    for (int i=1 ; i+ len-1 <= n;i ++){
      int j= i+len-1;
      f[i][j] = 1e8;
      for (int k=i;k<j;k++){
        f[i][j] = min(f[i][j],f[i][k] + f[k + 1][j] + s[j] - s[i-1]);
      }
    }
  }
  cout << f[1][n] << endl;
}
```

## 最长公共子序列

题目：[https://www.acwing.com/problem/content/899/](https://www.acwing.com/problem/content/899/)

![image-20210530180404930](https://gitee.com/coronapolvo/images/raw/master/20210530180406image-20210530180404930.png)

`状态表示:` f(i,j)

1. 集合：所有A[1-i]与B[1-j]的公共子序列的集合
2. 属性：max

`状态计算:`

对于f(i,j) 需要分为4种情况；在求最大值/最小值的时候情况可以重复但是不可以遗漏；

```c++
#include <iostream>

using namespace std;

const int N = 1010;

int n,m;
char a[N],b[N];
int f[N][N];

int main(){
    cin >> n >> m >> a + 1 >> b + 1;
    
    for(int i = 1;i <= n;i++){
        for(int j =1; j<=m;j ++ ){
            f[i][j] = max(f[i-1][j],f[i][j-1]);
            if(a[i] == b[j]) f[i][j] = max(f[i][j],f[i-1][j-1]+1);
        }
    }
    cout << f[n][m] << endl;
    return 0;
}
```











































