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

#### [1.2 打家劫舍](https://leetcode.cn/problems/house-robber/)

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

## 1.3 最大子数组和（最大子段和）

#### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

- f[i] 的含义：以 i 结尾的最大子数组和
- DP 属性：Max
- 状态转移方程：f[i] = max(f[i - 1] + nums[i], nums[i])

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = nums[0], f = nums[0];
        for (int i = 1; i < nums.size();i ++)
        {
            f = max(f, 0) + nums[i];
            res = max(res, f);
        }
        return res;
    }
};
```

#### [2606. 找到最大开销的子字符串](https://leetcode.cn/problems/find-the-substring-with-maximum-cost/description/)

这题和上面的最大子数组和问题基本一样，只不过这次的数组是一个字符串，而且每个元素都有一个权值。

- f[i] 的含义：以 i 结尾的最大子数组和
- DP 属性：Max
- 状态转移方程：f[i] = max(f[i - 1] + s[i] 的 value, s[i] 的 value)

```cpp
class Solution {
public:
    int maximumCostSubstring(string s, string chars, vector<int>& vals) {
        unordered_map<char, int> hash;
        for (int i = 0;i < chars.size();i ++) hash[chars[i]] = i;

        s = ' ' + s;
        int res = 0, f = INT_MIN;
        for (int i = 1;i < s.size();i ++)
        {
            int value = s[i] - 'a' + 1;
            if (hash.count(s[i]))
                value = vals[hash[s[i]]];
            f = max(f, 0) + value;
            res = max(res, f);
        }
        return res;
    }
};
```


#### [1749. 任意子数组和的绝对值的最大值](https://leetcode.cn/problems/maximum-absolute-sum-of-any-subarray/description/)

这题是思路感觉和打家劫舍2有点像，都是需要求俩种情况的最大值。

- pm 是 positive max，表示以 i 结尾的最大子数组和，nm 是 negative min，表示以 i 结尾的最小子数组和
- ps 是 positive sum，表示以 i 结尾的正数和，ns 是 negative sum，表示以 i 结尾的负数和

连续和肯定要么是正数和，要么是负数和，所以我们可以分别求出以 i 结尾的最大子数组和和最小子数组和，然后取最大值即可。


```cpp
class Solution {
public:
    int maxAbsoluteSum(vector<int>& nums) {
        int pm = 0, nm = 0;
        int ps = 0, ns = 0;

        for (auto a: nums)
        {
            ps += a;
            pm = max(pm, ps);
            ps = max(0, ps);
            ns += a;
            nm = min(nm, ns);
            ns = min(0, ns);
        }
        return max(pm, -nm);
    }
};
```

#### [1191. K 次串联后最大子数组之和](https://leetcode.cn/problems/k-concatenation-maximum-sum/description/)

```
typedef long long LL;

const int MOD = 1e9 + 7;

class Solution {
public:
    int kConcatenationMaxSum(vector<int>& arr, int k) {
        LL mx = 0, l = 0, r = 0, sum = 0, s = 0;
        for (int i = 0; i < arr.size(); i ++ ) {
            sum += arr[i];
            l = max(l, sum);
            s = max(s, 0ll) + arr[i];
            mx = max(mx, s);
            if (i + 1 == arr.size()) r = s;
        }

        if (k == 1) return mx % MOD;
        if (sum < 0) return max(mx, l + r) % MOD;
        return max(sum * (LL)(k - 2) + l + r, mx) % MOD;
    }
};

```

#### [918. 环形子数组的最大和](https://leetcode.cn/problems/maximum-sum-circular-subarray/description/)

:::tip

该题题解来自于 [LeetCode 官方题解](https://leetcode.cn/problems/maximum-sum-circular-subarray/solutions/2350660/huan-xing-zi-shu-zu-de-zui-da-he-by-leet-elou/)。

:::

本题为「53. 最大子数组和」的进阶版, 建议读者先完成该题之后, 再尝试解决本题。 

求解普通数组的最大子数组和是求解环形数组的最大子数组和问题的子集。设数组长度为 $n$, 下标从 0 开始, 在环形情况中，答案可能包括以下两种情况：

1. 构成最大子数组和的子数组为 $\operatorname{nums}[i: j]$, 包括 $n u m s[i]$到 nums $[j-1]$ 共 $j-i$ 个元素, 其中 $0 \leq i<j \leq n$ 。
2. 构成最大子数组和的子数组为 $n u m s[0: i]$ 和 $n u m s[j: n]$ , 其中 $0<i<j<n$ 。

![picture 0](images/6ea2fd519a286539b620f97e5b0302c58a7979f34d02765d0c647e1dcab14797.png)  

第一种情况可以通过「53. 最大子数组和」的方法求解，第二种情况可以通过求出最小子数组和，然后用总和减去最小子数组和求出最大子数组和。

```cpp
class Solution {
public:
    int maxSubarraySumCircular(vector<int>& nums) {
        int n = nums.size();
        int pre_max = nums[0], max_res = nums[0];
        int pre_min = nums[0], min_res = nums[0];
        int sum = nums[0];

        for (int i = 1;i < n; i++)
        {
            pre_max = max(pre_max + nums[i], nums[i]);
            max_res = max(max_res, pre_max);

            pre_min = min(pre_min + nums[i], nums[i]);
            min_res = min(min_res, pre_min);
            sum += nums[i];
        }
        if (max_res < 0)
            return max_res;
        else
            return max(max_res, sum - min_res);
    }
};
```

#### [2321.拼接数组的最大分数](https://leetcode.cn/problems/maximum-score-of-spliced-array/description/)

这题难就难在如何把这个问题转化成一个最大子数组和问题。根据题目的要求我们可以知道，需要让最后的数组累加和最大，也就是将两个数组 nums1 和 nums2 的其中一段做互换的累加和最大。

假设 sum1 是 nums1 的累加和，sum2 是 nums2 的累加和。如果我们想要交换后的 sum1 最大，那么我们需要找到一个子区间，假设这个区间是 [i, j]，`sum1 - sum1[i:j] + sum2[i:j]` 是最大的，且大于 `sum2 - sum2[i:j] + sum1[i:j]`。

也就是说我们需要找到最大的 `sum2[i:j] - sum1[i:j]`，这个问题就转化成了找到一个区间使得区间和最大的问题。这个思路是不是非常巧妙呢？当然了由于最大的可能是 sum1 也有可能是 sum2，所以我们需要分别求出这俩种情况的最大值最后取最大值即可。


```cpp
typedef long long LL;

class Solution {
public:
    int maximumsSplicedArray(vector<int>& nums1, vector<int>& nums2) {
        LL sum1 = accumulate(nums1.begin(), nums1.end(), 0);
        LL sum2 = accumulate(nums2.begin(), nums2.end(), 0);

        LL max_diff_1 = nums2[0] - nums1[0], f1 = nums2[0] - nums1[0];
        LL max_diff_2 = nums1[0] - nums2[0], f2 = nums1[0] - nums2[0];
        int n = nums1.size();

        for (int i = 1;i < n;i ++)
        {
            LL a = nums2[i] - nums1[i], b = nums1[i] - nums2[i];
            f1 = max(f1, (LL)0) + a;
            f2 = max(f2, (LL)0) + b;
            max_diff_1 = max(max_diff_1, f1);
            max_diff_2 = max(max_diff_2, f2);
        }

        return max(sum1 + max_diff_1, sum2 + max_diff_2);
    }
};
```

#### [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/description/)

这题一看还以为可以直接套用最大子数组和的思路，但是其实不行的。因为乘积有正负号的影响，所以我们需要维护俩个数组，一个是最大乘积数组，一个是最小乘积数组，而且转移方程还挺复杂的。

对于 max_f[i]，我们可以有三种选择：

1. max_f[i - 1] * nums[i]: 说明前面的最大乘积数组乘以当前的数是最大的
2. min_f[i - 1] * nums[i]: 说明前面的最小乘积数组乘以当前的数是最大的
3. nums[i]: 说明当前的数是最大的

对于 min_f[i]，我们也是类似的：

1. max_f[i - 1] * nums[i]: 说明前面的最大乘积数组乘以当前的数是最小的
2. min_f[i - 1] * nums[i]: 说明前面的最小乘积数组乘以当前的数是最小的
3. nums[i]: 说明当前的数是最小的

```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int n = nums.size();
        vector<int> max_f(n), min_f(n);
        int res = nums[0];
        max_f[0] = nums[0], min_f[0] = nums[0];
        for (int i = 1;i < n;i ++)
        {
            max_f[i] = max(max_f[i - 1] * nums[i], max(nums[i], min_f[i - 1] * nums[i]));
            min_f[i] = min(min_f[i - 1] * nums[i], min(nums[i], max_f[i - 1] * nums[i]));
            res = max(res, max_f[i]);
        }
        return res;
    }
};
```

## 网格 DP

对于一些二维 DP（例如背包、最长公共子序列），如果把 DP 矩阵画出来，其实状态转移可以视作在网格图上的移动。

### 2.1 基础网格 DP

#### [LCR 166.珠宝的最高价值](https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/
description/)

这题就非常简单了，就是一个二维 DP 的问题，每次只能向下或者向右移动，求最大值即可。

```cpp
class Solution {
public:
    int jewelleryValue(vector<vector<int>>& frame) {
        int m = frame.size(), n = frame[0].size();
        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        
        for (int i = 1;i <= m;i ++)
        {
            for (int j = 1;j <= n;j ++)
            {
                f[i][j] = max(f[i - 1][j], f[i][j - 1]) + frame[i - 1][j - 1];
            }
        }
        return f[m][n];
    }
};
```

#### [62.不同路径](https://leetcode.cn/problems/unique-paths/description/)

这题的代码和上面的题目基本一样，只不过支持 DP 的属性是 Count。

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int> > f(m + 1, vector<int>(n + 1, 1));

        for (int i = 1;i < m;i ++)
            for (int j = 1;j < n;j ++)
                f[i][j] = f[i - 1][j] + f[i][j - 1];
        return f[m - 1][n - 1];
    }
};
```

#### [63.不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/description/)

这题在上一题的基础上增加了障碍物，我们需要在 DP 的过程中判断当前位置是否是障碍物，如果是的话就直接跳过。

```cpp
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& o) {
        int m = o.size(), n = o[0].size();
    
        vector<vector<int> > f(m, vector<int>(n, 0));

        f[0][0] = !o[0][0];
        for (int i = 0;i < m;i ++)
        {
            for (int j = 0;j < n;j ++)
            {
                if (o[i][j]) continue;
                if (i && !o[i - 1][j]) f[i][j] += f[i - 1][j];
                if (j && !o[i][j - 1]) f[i][j] += f[i][j - 1];
            }
        }
        
        return f[m - 1][n - 1];
    }
};
```

#### [64.最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/description/)

这题和上面的题目基本一样，只不过这次 DP 的属性是 Min。不过需要注意初始化的时候，第一行和第一列的初始化。

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int> > f(m + 1, vector<int>(n + 1));

        f[0][0] = grid[0][0];
        for (int i = 1;i < m;i ++) f[i][0] = f[i - 1][0] + grid[i][0];
        for (int j = 1;j < n;j ++) f[0][j] = f[0][j - 1] + grid[0][j];
        for (int i = 1;i < m;i ++)
            for (int j = 1;j < n;j ++)
                f[i][j] = min(f[i - 1][j], f[i][j - 1]) + grid[i][j];
        return f[m - 1][n - 1];
    }
};
```

#### [120.三角形最小路径和](https://leetcode-cn.com/problems/triangle/description/)

这题和上面的题目基本一样，只不过换成了三角形，注意好边界条件即可。

```cpp
class Solution {
public:
    int minimumTotal(vector<vector<int>>& t) {
        int m = t.size();
        vector<vector<int> > f(m, vector<int>(m, INT_MAX));
        f[0][0] = t[0][0];
        for (int i = 1;i < m;i ++) f[i][0] = f[i - 1][0] + t[i][0];
        
        for (int i = 1;i < m;i ++)
        {
            for (int j = 1;j <= i;j ++)
            {
                f[i][j] = min(f[i - 1][j - 1], f[i - 1][j]) + t[i][j];
            }   
        }

        int res = INT_MAX;
        for (int i = 0;i < m;i ++)
            res = min(res, f[m - 1][i]);
    
        return res;
    }
};
```

#### [931.下降路径最小和](https://leetcode-cn.com/problems/minimum-falling-path-sum/description/)

和上一题雷同，注意边界条件即可；

```cpp
class Solution {
public:
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int> > f(m + 1, vector<int>(n + 1));

        for (int i = 0;i < n;i ++) f[0][i] = matrix[0][i];
        for (int i = 1;i < m;i ++)
        {
            for (int j = 0;j < n;j ++)
            {
                int a = f[i - 1][j];
                if (j) a = min(f[i - 1][j - 1], a);
                if (j < n - 1) a = min(f[i - 1][j + 1], a);
                f[i][j] = a + matrix[i][j];
            }
        }
        int res = INT_MAX;
        for (int i = 0;i < n;i ++)
            res = min(res, f[m - 1][i]);
        return res;
    }
};
```

### 2.2 进阶

#### [1594.矩阵的最大非负积](https://leetcode.cn/problems/maximum-non-negative-product-in-a-matrix/description/)

当需要处理负数的时候，我们可以维护俩个 DP 数组，一个是最大值，一个是最小值，这样就可以处理负数的情况了。这样的策略我们在上面的乘积最大子数组中也有提到。

```cpp
class Solution {
public:
    int maxProductPath(vector<vector<int>>& grid) 
    {
        typedef long long LL;
        const int MOD = 1e9 + 7;
        int m = grid.size(), n = grid[0].size();
        vector<vector<LL> > maxgt(m, vector<LL>(n));
        vector<vector<LL> > minlt(m, vector<LL>(n));

        maxgt[0][0] = minlt[0][0] = grid[0][0];
        for (int i = 1;i < n;i ++)
            maxgt[0][i] = minlt[0][i] = maxgt[0][i - 1] * grid[0][i];
        for (int i = 1;i < m;i ++)
            maxgt[i][0] = minlt[i][0] = maxgt[i - 1][0] * grid[i][0];
        
        for (int i = 1;i < m;i ++)
        {
            for (int j = 1;j < n;j ++)
            {
                if (grid[i][j] >= 0)
                {
                    maxgt[i][j] = max(maxgt[i][j - 1], maxgt[i - 1][j]) * grid[i][j];
                    minlt[i][j] = min(minlt[i][j - 1], minlt[i - 1][j]) * grid[i][j];
                } else 
                {
                    maxgt[i][j] = min(minlt[i][j - 1], minlt[i - 1][j]) * grid[i][j];
                    minlt[i][j] = max(maxgt[i][j - 1], maxgt[i - 1][j]) * grid[i][j];
                }
            }
        }

        if (maxgt[m - 1][n - 1] < 0) return -1;
        return maxgt[m - 1][n - 1] % MOD;
    }
};
```

#### [2435.矩阵中和能被K整除的路径](https://leetcode.cn/problems/paths-in-matrix-whose-sum-is-divisible-by-k/description/)

f[i][j][v] 表示从 (0, 0) 到 (i, j) 的路径和模 k 为 v 的方案数。

考虑到是从左上角转移过来的，所以我们可以直接枚举上一个状态的 v，然后转移即可。

f[i][j][(v + grid[i][j]) % k] = f[i - 1][j][v] + f[i][j - 1][v];

这里我为了少一些边界判断，所以我直接把 f 的大小设置为 m + 1 和 n + 1，而且在遍历的时候每次计算 f[i+1][j+1][v] 的时候，这样就可以少一些边界判断。

需要注意的是初始化的时候，我们写的是 f[1][0][0] = 1 或者 f[0][1][0] = 1，因为的 f 是从 1 开始的，所以我们需要初始化 f[1][0][0] 而不是 f[0][0][0]。

```cpp
class Solution {
public:
    int numberOfPaths(vector<vector<int>>& grid, int k) {
        const int MOD = 1e9 + 7;
        int m = grid.size(), n = grid[0].size(), f[m + 1][n + 1][51];
        memset(f, 0, sizeof(f));

        f[1][0][0] = 1;
        for (int i = 0;i < m;i ++)
        {
            for (int j = 0;j < n;j ++)
            {
                for (int v = 0; v < k; v ++)
                    f[i + 1][j + 1][(v + grid[i][j]) % k] = (f[i + 1][j][v] + f[i][j + 1][v]) % MOD;
            }
        }

        return f[m][n][0];
    }
};
```

#### [174.地下城游戏](https://leetcode.cn/problems/dungeon-game/)

:::tip

该题题解来自于 [LeetCode 官方题解 174.地下城游戏](https://leetcode.cn/problems/dungeon-game/solutions/326171/di-xia-cheng-you-xi-by-leetcode-solution/)

:::

这道题的题目一看上去就像是一个 DP 的题目，但是只要你仔细思考就会发现，如果你从左上角开始走，那么我们需要同时记录两个值。第一个是「从出发点到当前点的路径和」，第二个是「从出发点到当前点所需的最小初始值」。而这两个值的重要程度相同。

于是我们考虑从右下往左上进行动态规划。令 f[i][j] 表示从 (i, j) 到终点所需的最小初始值。换句话说，当我们到达坐标 (i,j) 的时候，我们至少需要 f[i][j] 的血量才能走到终点。

这样一来，我们就无需担心路径和的问题，只需要关注最小初始值。我们可以从右下角开始，然后逐步向左上角推导出 f[0][0] 即可。

f[i][j] 是从 f[i+1][j] 和 f[i][j+1] 中选择一个最小值，然后减去 dungeon[i][j]，但是如果减去 dungeon[i][j] 之后的值小于等于 0，那么我们就需要设置为 1。因为我们至少需要 1 的血量。

边界条件是，当 i = m - 1 或者 j = n - 1 的时候，我们需要特殊处理一下，因为这个时候我们无法选择 f[i+1][j] 或者 f[i][j+1]，因此代码实现中给无效值赋值为极大值。

```cpp
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int m = dungeon.size(), n = dungeon[0].size();
        vector<vector<int> > f(m + 1, vector<int>(n + 1, INT_MAX));
        f[m - 1][n] = f[m][n - 1] = 1;
        for (int i = m - 1;i >= 0;i --)
        {
            for (int j = n - 1;j >= 0;j --)
            {
                int mini = min(f[i + 1][j], f[i][j + 1]);
                f[i][j] = max(mini - dungeon[i][j], 1);
            }
        }
        return f[0][0];
    }
};
```

#### [741. 摘樱桃](https://leetcode.cn/problems/cherry-pickup/description/)

由于正向和反向的路可以被视为等价的，所以我们不妨把这个问题转化成有俩个人同时从 （0, 0）出发，同时走到（n - 1, n - 1），求俩个人摘到的樱桃数和的最大值。

我们用 k 表示走了多少步，第一个人的坐标是 (x1,y1), 第二个人的坐标是 (x2,y2)，当 x1 == x2 的时候，这个时候 y1 和 y2 也是相等的。

定义 $f[k]\left[x_1\right]\left[x_2\right]$ 表示两个人（设为 A 和 B）从 $(0, 0)$ 和 $\left(0, 0\right)$ 同时出发, 分别到达 (x1, k - x2) 和 (x2, k - x1) 摘到的樱桃个数之和的最大值。

1. 都走右边：f[k][x1][x2]
2. 一个人走右边，一个人走下边：f[k][x1][x2 - 1]
3. 一个人走下边，一个人走右边：f[k][x1 - 1][x2]
4. 都走下边：f[k][x1 - 1][x2 - 1]

最后的答案就是 f[2 * n - 2][n - 1][n - 1]。

```cpp
class Solution {
public:
    int cherryPickup(vector<vector<int>>& grid) {
        int n = grid.size();
        vector<vector<vector<int>>> f(n * 2 - 1, vector<vector<int>>(n, vector<int>(n, INT_MIN)));
        f[0][0][0] = grid[0][0];
        for (int k = 1;k < n * 2 - 1;k ++)
        {
            for (int x1 = max(k - n + 1, 0);x1 <= min(k, n - 1); x1++)
            {
                int y1 = k - x1;
                // 走到了障碍物
                if (grid[x1][y1] == -1) continue;
                for (int x2 = x1; x2 <= min(k, n - 1); x2 ++)
                {
                    int y2 = k - x2;
                    // 走到了障碍物
                    if (grid[x2][y2] == -1) continue;
                    int res = f[k - 1][x1][x2];
                    if (x1)
                        res = max(res, f[k - 1][x1 - 1][x2]);
                    if (x2)
                        res = max(res, f[k - 1][x1][x2 - 1]);
                    if (x1 && x2)
                        res = max(res, f[k - 1][x1 - 1][x2 - 1]);
                    res += grid[x1][y1];
                    // 避免俩个人摘到同一个樱桃
                    if (x1 != x2)
                        res += grid[x2][y2];
                    
                    f[k][x1][x2] = res;
                }
            }
        }
        return max(f[2 * n - 2][n - 1][n - 1], 0);
    }
};
```

## 3. 背包 问题

### 3.1 0-1 背包问题

物品只能取一次

#### [2915. 和为目标值的最长子序列的长度](https://leetcode.cn/problems/length-of-the-longest-subsequence-that-sums-to-target/)

恰好装满型 0-1 背包，这题属于模板题，如果问题是恰好装满型的话，我们可以把问题转化成一个背包问题，然后求出最大的长度即可。

和普通的 0-1 背包问题对比的话，我们就是外层循环 nums， 内层循环 target，然后求出最大的长度即可。

f[i] 的含义是和为 i 的最长子序列的长度。

它的状态转移方程是：f[i] = max(f[i], f[i - x] + 1)


```cpp
class Solution {
public:
    int lengthOfLongestSubsequence(vector<int>& nums, int target) {
        vector<int> f(target + 1, INT_MIN);
        f[0] = 0;
        int s = 0; // 记录前缀和
        for (int x: nums)
        {
            s = min(s + x, target); // 防止 s 超过 target
            for (int j = s;j >= x;j --)
            {
                f[j] = max(f[j], f[j - x] + 1);
            }
        }
        
        return f[target] > 0 ? f[target] : -1;
    }
};
```

#### [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/description/)

这题其实本质上和上一题一样，只不过这次的 target 是 sum / 2，题目可以转化为是否存在一个子集的合等于 sum / 2。

这里 f 的含义是，是否有一个子集的和等于 i。当然了你也可以用和上题一样的 f 含义。区别是初始化不同和转移方程不同。

```cpp
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for (int x: nums) sum += x;
        if (sum % 2) return false;

        vector<int> f(sum / 2 + 1);
        f[0] = 1;
        for (int x: nums)
        {
            for (int j = sum / 2;j >= x;j --)
            {
                f[j] = min(f[j] + f[j - x], 1);
            }
        }
        return f[sum / 2] > 0;
    }
};
```

#### [494. 目标和](https://leetcode.cn/problems/target-sum/description/)

这题涉及到负数，所以我们需要维护一个偏移量，然后把问题转化成一个 0-1 背包问题。所谓每个数可以取正号或者负号，其实就是一共 2n 个数，然后求和为 target 的方案数。

这里 f[i][j] 的含义是 nums 的前 i 个数，和为 j 的方案数。它的状态可以从 f[i - 1][j - x] 或者 f[i - 1][j + x] 转移过来。

```cpp
class Solution {
public:
    int findTargetSumWays(vector<int>& a, int S) {
        int n = a.size(), Offset = 1000;

        vector<vector<int>> f(n + 1, vector<int>(2010));

        f[0][Offset] = 1;
        for (int i = 1;i <= n;i ++)
        {
            for (int j = -1000;j <= 1000;j ++)
            {
                if (j - a[i - 1] >= -1000)
                    f[i][j + Offset] += f[i - 1][j - a[i - 1] + Offset];
                if (j + a[i - 1] <= 1000)
                    f[i][j + Offset] += f[i - 1][j + a[i - 1] + Offset];
            }
        }
        return f[n][S + Offset];
    }
};
```

#### [2787. 将一个数字表示成幂的和的方案数](https://leetcode.cn/problems/ways-to-express-an-integer-as-sum-of-powers/description/)

和上面的题目基本都是一模一样的，唯一的区别就是 a 需要自己计算一下。

这里 f 的含义是，将 i 表示成幂的和的方案数，属性是 Count。

```cpp
class Solution {
public:
    int numberOfWays(int n, int x) {
        typedef unsigned long long ULL;
        const int MOD = 1e9 + 7;
        vector<ULL> a;
        int m = 1;
        while (pow(m, x) <= n)
        {
            a.push_back(pow(m, x));
            m ++;
        }

        vector<ULL> f(n + 1);
        f[0] = (ULL)1;
        for (int i = 1;i <= a.size();i ++)
        {
            for (int j = n;j >= a[i - 1];j --)
            {
                f[j] += f[j - a[i - 1]];
            }
        }

        return f[n] % MOD;
    }
};
```


