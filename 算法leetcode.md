#### 计数问题

##### 最长平衡子字符串

求一个01字符串中最长平衡子字符串。平衡字符串为0开头与1同样长度的字符串。

```python
class Solution:
    def findTheLongestBalancedSubstring(self, s: str) -> int:
        res = 0
        n = len(s)
        count = [0, 0]
        for i in range(n):
            if s[i] == '1':
                count[1] += 1
                res = max(res, 2 * min(count))
            elif i == 0 or s[i-1] == '1':
                count[0] = 1
                count[1] = 0
            else:
                count[0] += 1
        return res
```



遍历一遍就可以。

- 两个计数器，可以判断在每一个位置时当前的平衡字符串0和1个数
- 如果是数到1后的0，代表着一个新的平衡字符串的开始，name要重启计数器



**要点**：

​	通过max函数实时更新答案



##### 分配糖果

相邻孩子评分**绝对高**的得到糖果更多，最低为1

方法1：

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        ans = [1] * len(ratings)
        for i in range(len(ratings) - 1):
            if ratings[i + 1] > ratings[i]:
                ans[i+1] = ans[i] + 1
        
        for i in range(len(ratings) - 1,0,-1):
            if ratings[i] < ratings[i-1]:
                tmp = ans[i] + 1
                print(tmp)
                ans[i - 1] = max(tmp,ans[i-1])

        return sum(ans)
```

左右遍历一遍，满足：

- 向右遍历时，右边的始终比左边大1
- 向左遍历时，左边的始终比右边大1，除非左边遇到了突变，这时要比较大小

方法2：

记录每个递增\递减序列的长度





##### 射气球

求一些最小数量的【钉子】，这些【钉子】可以【钉住】所有的目标【区间】