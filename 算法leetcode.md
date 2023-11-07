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