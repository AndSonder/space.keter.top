# Leetcode DP 专题

## 1. 入门 DP

### 1.1 爬楼梯

#### [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

非常经典的爬楼梯问题，可以使用动态规划解决。

```cpp
class Solution {
public:
    int climbStairs(int n) {
        int a = 1, b = 1, c;
        while (-- n)
        {
            c = a + b;
            a = b;
            b = c;
        }
        return b;
    }
};
```

#### [746. 使用最小花费爬楼梯](https://leetcode.cn/problems/min-cost-climbing-stairs/)

在爬楼梯的基础上，增加了花费的限制。

- f[i] 的含义：爬到第 i 级楼梯的最小花费
- DP 属性：Min
- 状态转移方程：f[i] = min(f[i - 1] + cost[i - 1], f[i - 2] + cost[i - 2])

```cpp
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        vector<int> f(n + 1);

        f[0] = 0;
        f[1] = 0;

        for (int i = 2;i <= n;i ++)
        {
            f[i] = min(f[i - 1] + cost[i - 1], f[i - 2] + cost[i - 2]);
        }
        return f[n];
    }
};
```

#### [377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/)

- f[i] 的含义：和为 i 的方案数
- DP 属性：Count
- 状态转移方程：f[i] = sum(f[i - nums[j]])

```cpp
class Solution {
public:
    int combinationSum4(vector<int>& nums, int m) {
        vector<unsigned> f(m + 1);
        f[0] = 1;
        for (int i = 0;i <= m;i ++)
        {
            for (auto j: nums)
            {
                if (i >= j)
                    f[i] += f[i - j];
            }
        }
        return f[m];
    }
};
```

#### [2466. 统计构造好字符串的方案数](https://leetcode.cn/problems/count-ways-to-build-good-strings/)

这题不要被题目吓到，其实就是一个爬楼梯问题。

- f[i] 的含义：长度为 i 的字符串的方案数
- DP 属性：Count
- 状态转移方程：f[i] = f[i - one] + f[i - zero]

```cpp
class Solution {
public:
    int countGoodStrings(int low, int high, int zero, int one) {
        vector<int> f(high + 1);
        const int M = 1e9 + 7;

        f[0] = 1;
        int ans = 0;
        for (int i = 1;i <= high;i ++)
        {
            if (i >= one) f[i] = (f[i] + f[i - one]) % M;
            if (i >= zero) f[i] = (f[i] + f[i - zero]) % M;
            if (i >= low) ans = (ans + f[i]) % M;
        }
        return ans;
    }
};
```

#### [2266. 统计打字方案数](https://leetcode.cn/problems/count-number-of-texts/)

这题也是一个爬楼梯问题。每一个子串都可以用爬楼问题求解出来一个方案数，最后所有的子串的方案数相乘即可。和上面那个组合总和问题类似。

- f[i] 的含义：长度为 i 的字符串的方案数
- DP 属性：Count
- 状态转移方程：f[i] = sum(f[i - j])

```cpp
class Solution {
public:
    int countTexts(string pressedKeys) {
        typedef unsigned long long ULL;
        const int MOD = 1e9 + 7;
        int a[10] = {0, 0, 3, 3, 3, 3, 3, 4, 3, 4};

        int n = pressedKeys.size();
        int ans = 1;
        int l = 0, r = 0;
        for (int i = 1;i <= n;i ++)
        {
            if (i == n || pressedKeys[i] != pressedKeys[i - 1])
            {
                r = i - 1;
                int m = r - l + 1;
                if (m > 1) 
                {
                    vector<ULL> f(m + 1);
                    f[0] = 1, f[1] = 1;
                    for (int i = 2;i <= m;i ++)
                    {
                        for (int j = 1;j <= a[pressedKeys[l] - '0'];j ++)
                        {
                            if (i >= j)
                                f[i] = (f[i] + f[i - j]) % MOD;
                        }
                    }
                    ans = (ans * f[m]) % MOD;
                }
                l = i;
            }
        }
        return ans;
    }
};
```

### [1.2 打家劫舍](https://leetcode.cn/problems/house-robber/)

- f[i] 的含义：偷窃第 i 个房子的最大收益
- DP 属性：Max
- 状态转移方程：f[i] = max(f[i - 1], f[i - 2] + nums[i])

```cpp
class Solution {
public:

    int rob(vector<int>& nums) 
    {
        int n = nums.size();
        vector<int> f(n + 1);

        if (!n) return 0;
        f[0] = nums[0];
        if (n < 2)
            return f[0];
        f[1] = max(nums[0], nums[1]);

        for (int i = 2;i < n; i++)
        {
            f[i] = max(f[i - 1], f[i - 2] + nums[i]);
        }
        return f[n - 1];
    }
};
```

### [740. 删除并获得点数](https://leetcode.cn/problems/delete-and-earn/description/)

- f[i][0] 的含义：不选择第 i 个数的最大收益， f[i][1] 的含义：选择第 i 个数的最大收益
- DP 属性：Max
- 状态转移方程：f[i][0] = max(f[i - 1][0], f[i - 1][1]), f[i][1] = f[i - 1][0] + i * cnt[i]

```cpp
const int N = 1e4 + 10;
int cnt[N];
int f[N][2];

class Solution {
public:
    int deleteAndEarn(vector<int>& nums) {
        memset(cnt, 0, sizeof(cnt));
        memset(f, 0, sizeof(f));

        for (int i = 0;i < nums.size();i ++) cnt[nums[i]] ++;

        int res = 0;
        for (int i = 1;i < N;i ++)
        {
            f[i][0] = max(f[i - 1][0], f[i - 1][1]);
            f[i][1] = f[i - 1][0] + i * cnt[i];
            res = max(res, f[i][1]);
        }

        return res;
    }
};
```

### [2320. 统计放置房子的方式数](https://leetcode.cn/problems/count-number-of-ways-to-place-houses/description/)

这题俩侧的房子可以分开看，最后排列组合（方案数相乘）即可。

- f[i] 的含义：在 0 - i 的位置放置房子的方案数
- DP 属性：Count
- 状态转移方程：f[i] = f[i - 1] + f[i - 2]

因为房子不能相邻，所以在 i 位置放置房子的方案数等于 i - 2 位置放置房子的方案数加上 i - 1 位置放置房子的方案数，其实就是一个爬楼梯问题换了个皮。

```cpp
class Solution {
public:
    int countHousePlacements(int n) {
        const int MOD = 1e9 + 7;
        vector<unsigned long long> f(n + 1);
        f[0] = 1;
        f[1] = 2;
        for (int i = 2;i <= n;i ++)
            f[i] = (f[i - 1] + f[i - 2]) % MOD;
        return f[n] * f[n] % MOD;
    }
};
```

### [打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/)

这题和上面的打家劫舍问题类似，只不过这次是一个环形的房子，所以我们可以分成两种情况来讨论：

- 偷第一个房子，不偷最后一个房子
- 不偷第一个房子，偷最后一个房子

俩种情况取最大值即可。

- f[i] 的含义：偷窃第 i 个房子的最大收益
- g[i] 的含义：不偷窃第 i 个房子的最大收益
- DP 属性：Max
- 状态转移方程：f[i] = g[i - 1] + nums[i - 1], g[i] = max(f[i - 1], g[i - 1])


```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        // f[i] 表示偷窃第 i 个房子的最大收益
        // g[i] 表示不偷窃第 i 个房子的最大收益
        vector<int> f(n + 1), g(n + 1);
        if (!n) return 0;
        if (n == 1) return nums[0];

        // 不偷第一个房子
        for (int i = 2; i <= n;i ++)
        {
            f[i] = g[i - 1] + nums[i - 1];
            g[i] = max(f[i - 1], g[i - 1]);
        }

        int res = max(g[n], f[n]);

        f[1] = nums[0];
        g[1] = INT_MIN;

        // 偷第一个房子
        for (int i = 2; i <= n;i ++)
        {
            f[i] = g[i - 1] + nums[i - 1];
            g[i] = max(f[i - 1], g[i - 1]);
        }
        return max(res, g[n]);
    }
};
```