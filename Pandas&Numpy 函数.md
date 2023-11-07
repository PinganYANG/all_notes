#### Pandas 函数

##### groupby

groupby用关键字`by=[keys]`确定要groupby的关键字。接着后面可以接上`.agg({'col':'min'})`或`.min()`这一类的聚合函数。

并且要想同时对col使用不同的agg函数时，要用`.agg({'col':['min','max','count']})`

然后是`.groupby(by = ['email'])['id'].min()`表示在email上用groupby，然后在id上作用聚合函数，但这里的返回值是行key为email 值为min值的可以直接`to_dict()`的部分。**返回的是一个series，并且将email（groupby的key）作为index，因此可以用reset_index 创造新列**

但如果将聚合函数`.min()`换为`.transform('min')`，则可以直接加上一列，将min的值附加到这一列之中。

###### .agg 同时对多个列使用聚合函数

` df.groupby('group_col').agg({'col1': 'sum', 'col2': 'count'})`

##### drop

drop里面用columns=list 或者index=list，其中要删除df中的一个子块时，可以在里面放 sub_df.index `person.drop(removed_person.index, inplace=True)`

利用`inplace=True`控制是否在原地改变

##### dropna

df.dropna(axis='columns'/'index',how='any'任意'all'全部,subset=[指定的列],inplace=)

##### sort_values

by控制keys

用ascending控制升序降序

```python
scores = scores.sort_values(by=['rank'])
```

返回的是.sort_values前这一部分排序后的值。比如如果是scores['rank'].sort_values，那么就会返回对应rank的series。这样要对一个df排序，那么就直接对df整个sort_values

###### 多排序

我们观察到，**by后的key其实是一个list**，因此我们可以通过by=list和ascending=list来同时升序降序，以前序为主

```python
sorted_df = df.sort_values(by=['column1', 'column2'], ascending=[True, False])
```

###### 多级排序

sort_values(by=(0级索引，1级索引...))



##### join

```python
df.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)
```

一般join用于index的join

如果想用on=col的话还是直接用merge吧



##### rank

用法：

- df[col].rank(axis=0, method='average', numeric_only=False, na_option='keep', ascending=True, pct=False)
  - method中有不同的用法
    - max 表示 1 3 3 4， 有两个相同的时都取最大的
    - min 表示 1 2 2 4， 有两个相同的取最小的
    - average 表示 1 2.5 2.5 4， 有两个相同的取均值
    - dense 表示 1 2 2 3，有两个相同的取小的并且不占位
    - 默认average
  - pct可以输出排序的百分位



```python
merged_df['rank'] = merged_df.groupby('departmentId')['salary'].rank(method='dense', ascending=False) 
```

巧用dense rank和groupby直接在原表中加上分组rank



##### 处理datetime 比如早于 晚于 之间

早于就用 >

晚于就用 <

在之间就用 between



确定一个datetime可以用datetime.datetime(yyyy,mm,dd)



##### Round的用法

df.round(2)

可以round所有数，保留两位小数

df['col'].round(2)

则是round这一列的数，并保留两位小数

**【有一个傻逼的点，就是他的0.125会舍入到0.12，要用内置的demical来四舍五入】**

##### concat

使用 pd.concat 时，axis 1 是列合并【1就是竖着的所以是列】



##### pivot 【多级索引】【sort_values】

df.pivot(index='name',columns='year',values='gdp') 这样可以将columns列的category值展成新列名，值为gdp，rows的index为name

pivot 后会形成多级索引，从上到下/从左到右 依次是 0,...,n级索引

- 多级排序：sort_values(by=(0级索引，1级索引...))
- 生成多级df pd.DataFrame({(0级索引，1级索引):[list]})



##### idxmax & idxmin

idxmax是取一个系列中最大值的index。

可以通过

```python
# 找到小于给定值的最大值的索引
index = data[data < given_value].idxmax()

# 找到大于给定值的最小值的索引
index = data[data > given_value].idxmin()
```



##### apply 

```python
df['E'] = df.apply(lambda row: row['A'] * row['B'] if row['C'] > 10 else 0, axis=1)
```

在 **:** 后面可以加上函数内容



##### fillna

- 固定值填充
  - df.fillna(0)
- 向前填充
  - df.fillna(method='**ffill**')
- 向前填充
  - df.fillna(method='**bfill**')



##### rolling

滑动窗口函数

df[col].rolling(window=n).聚合函数/apply自定义函数

可以实现pandas的时间窗操作



##### cumsum

cumsum是可以用在连续问题中的重要函数。主要用到了cumsum可以对bool型列的特殊处理。

- 首先cumsum会对True取1，False取0
- 所以当一列中仅有True和False时，cumsum就可以以True为截断进行组号划分，因为遇到True时才会加一，而一直是false时则组号回一直不变，详情用法见**连续问题**

##### cumcount

就是一个累计计数的函数





#### Numpy 函数

##### np.sort()

默认升序排列，且不能通过设置倒序，但可以用np.sort()[::-1]来倒排



##### sorted

排序，可以用关键字reverse控制正序和倒序

##### np.flatten

展平函数



##### .nunique 【groupby 的 agg 函数，相当于count(distinct)】





##### select

```python
conditions = [
    (df['col1'] > df['col2']).astype(np.bool_),  # 条件1
    (df['col1'] == df['col2']).astype(np.bool_), # 条件2
    (df['col1'] < df['col2']).astype(np.bool_)   # 条件3
]

# 对应的值
choices = [3, 1, 0]

df['new_col'] = np.select(conditions, choices, default=np.nan)
```

可以进行列对比多条件赋值，针对**两列比较多返回值**时有奇效

- 注意要用bool_来改变dtype，否则会报错【很怪】





![image-20231027201854746](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231027201854746.png)



#### 连续问题

##### 连续固定天

首先可以用lambda函数直接取对应值

可以用rolling函数取窗口



##### 满足条件的连续不固定天

假设**需要的条件**：

- 连续三天
- people至少100



- 那么先为满足value条件的设置flag
  - stadium['flag'] = stadium['people'] >= 100

- 再用到cumsum和flag来为满足连续的分组
  - group_ids = (~stadium['flag']).cumsum()
  - grouped = stadium.groupby(group_ids)

- 最后再取满足连续三天条件的

  - result = [group for _, group in grouped if len(group) > 3]

    ans = [a.loc[a['flag'],['id','visit_date','people']] for a in result]

- 最后concat到一起

```python
def human_traffic(stadium: pd.DataFrame) -> pd.DataFrame:
    stadium['flag'] = stadium['people'] >= 100

    group_ids = (~stadium['flag']).cumsum()

    grouped = stadium.groupby(group_ids)

    result = [group for _, group in grouped if len(group) > 3]
    
    ans = [a.loc[a['flag'],['id','visit_date','people']] for a in result]

    if len(ans) == 0:
        stadium.drop(columns = ['flag'],inplace=True)
        return stadium.drop(index=stadium.index)
    else:
        return pd.concat(ans)
```
