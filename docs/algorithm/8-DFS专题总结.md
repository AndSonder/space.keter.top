# DFS专题总结

## 排列问题
### 全排列
:::tip
首先对nums进行排列，使用bool记录该位是否使用过，循环遍历数组每一位
:::

```c++
vector<bool> st(nums.size(),false);
vector<int> path(nums.size()); 
vector<vector<int>> ans;
// e.g. nums = [1,1,2]
// u 用来记录排列到哪一位了
void dfs(vector<int> nums, int u,int start)
{
	if(u  == nums.size())
	{
		ans.push_back(path);
		return;
	}

	for(int i = start;i < nums.size();i++)
	{
		// 如果这位还没有使用过
		if(!st[i])
		{
			// 状态修改
			st[i] = true;
			path[i] = nums[i];
			if(u + 1 < nums.size() and nums[u+1] != nums[u])
				dfs(nums,u+1,0); // 虽然是从0开始排列的当时st[i]可以帮助不重复排列
			else
				dfs(nums,u+1,i+1);
			st[i] = false; // 状态复原
		}
	}
}
```

### 组合总和
:::tip
给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

思路分析：对于每一次递归都有两种情况，一种是选取这一位，一种是不选取这一位；
:::

```c++
class Solution {
public:
    vector<int> path; // 存储路径
    vector<vector<int>> ans; // 存储答案
    
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        dfs(candidates, 0, target);
        return ans;
    }

    void dfs(vector<int> &candidates, int u, int target)
    {
		// 达到目标停止递归
        if(target == 0)
        {
            ans.push_back(path);
            return;
        }
		// 和的输入超过target了，停止递归
        if(target < 0) return;
        if(u == candidates.size()) return;
        // 不选取自己
        dfs(candidates,u+1,target);
        // 选取自己
        path.push_back(candidates[u]);
        dfs(candidates,u,target-candidates[u]);
        path.pop_back();
    }
};
```

### 组合总和 II
:::tip
给定一个候选人编号的集合 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

思路：关键在于递归的开始点如何计算
:::

```c++
class Solution {
public:

    vector<vector<int>> ans; // 存储答案
    vector<int> path; // 存储路径

    vector<vector<int>> combinationSum2(vector<int>& c, int target) {
        sort(c.begin(), c.end());
        dfs(c, 0, target);

        return ans;
    }

    void dfs(vector<int>& c, int u, int target) {
        if (target == 0) {
            ans.push_back(path);
            return;
        }
        if (u == c.size()) return;

		// 寻找下一个不重复数
        int k = u+1;
		while(k < c.size() and c[k] == c[u]) k++;
		int cnt = u - k;
		for(int i = 0;c[u] * i <= target and i <= cnt;i++)
		{
			dfs(c,k,target-c[u] * i);
			path.push_back(c[u]); // 状态修改
		}
		for(int i = 0;c[u] * i <= target and i <= cnt;i++)
			path.pop_back(); // 状态复原
    }
};
```


