#### 赋值

x <- 5 

- 箭头就是赋值方向
- typeof(x) 判断类型
- length(x) 判断变量长度

#### 向量

用c(1,2,3,4,5...) 来创建向量，并用来赋值

x <- c(1,2,3,4,5)

##### 向量命名

```R
x <- c('a' = 5, 'b' = 6, 'c' = 7, 'd' = 8)
```

或

```R
x <- c(5, 6, 7, 8)
names(x) <- c('a', 'b', 'c', 'd')
```

来为向量的列命名

##### 向量运算

两向量加减乘除要在对应的位置进行运算

- 长度相同：直接对应
- 长度不同：短的循环至与长的相同

##### seq

生成等差数列，【与Python [1:10:2]思想类似】

```R
s1 <- seq(from = 0, to = 10, by = 0.5)
```

如果by是0的话，可以直接from:to

```R
s4 <- 0:10  # Colon operator (with by = 1):
```

##### rep

重复复制

```R
s2 <- rep(x = c(0, 1), times = 3)
```

##### 因子型向量

```r
gender <- factor(c("male", "female", "female", "male"))
```

因子型类比于enum类，枚举类，用于一些离散变量和因子分析

#### 矩阵

```R
matrix(
  c(2, 4, 3, 1, 5, 7),
  nrow = 2, 
  ncol = 3,
  byrow = TRUE
)
```

用matrix创建矩阵，其中传入的是一个一维向量，byrow指定是否先填充row

#### 列表

```R
list1 <- list(
  a = c(5, 10),
  b = c("I", "love", "R", "language", "!"),
  c = c(TRUE, TRUE, FALSE, TRUE)
)
```

将一些向量储存在一起

#### 数据框

```R
df <- data.frame(
  name      = c("Alice", "Bob", "Carl", "Dave"),
  age       = c(23, 34, 23, 25),
  marriage  = c(TRUE, FALSE, TRUE, FALSE),
  color     = c("red", "blue", "orange", "purple")
)
```

相当于Python df

#### 函数

##### 内置函数

sum

print

sqrt

mean

sd

min

max

length

sort

unique

quantile

is.numeric

as.character

as.logical

as.numeric

ifelse(条件，对的，不对的)

##### 自定义函数

```R
my_std <- function(x) {
   (x - mean(x)) / sd(x)
 }
```



如果return的值是需要返回最后一行的值，那么return可以省略

#### 条件语句

```R
    if (is.numeric(num)) {
      num^2
    } else {
     "Your input is not numeric."
    }
```

形式与JS挺像的

#### 切片

##### 对于向量

- 正整数 相当于index
- 负整数 排除对应的位置的参数
- 条件选择 x[x>2] 与Numpy类似

##### 对于列表list



##### 对于df

类似pandas

一般都用df处理一些结构性数据

#### df数据处理

##### mutate 新增一列

mutate(.data = df, name = value)



##### 管道 %>%

value %>% 函数（需要value的地方）

用管道将一个value传入到一个需要data的函数之中

##### select（）

- 列名
- 整数：列索引
- 条件:
  - ends_with()
  - contains()
  - where(is.character)
  - stars_with()

##### rename()

rename(.data, new_name = old_name)

##### filter()

filter(列条件)【类似df.loc[]】

##### summarise（）

统计汇总

##### group_by()

group_by要和summarise一起用，相当于dataframe的聚合函数

```R
df_new %>%
  group_by(name) %>%
  summarise(
    mean_score = mean(total),
    sd_score   = sd(total)
  )
```

##### arrange()排序

arrange(name) 按name列排序

- 如果要倒序加上desc
- arrange(desc(name))

多维排序：

- 加上两列 

- 

- ```R
  df_new %>% 
    arrange(type, desc(total))
  ```

##### left_join()

```R
left_join(df1, df2, by = "name")
```

##### right_join()

```R
left_join(df1, df2, by = "name")
```

##### full_join()

相当于outer_join()

##### inner_join()

##### semi_join()

- 半联结`semi_join(x, y)`，保留name与df2的name相一致的所有行

##### anti_join()

- 反联结`anti_join(x, y)`，丢弃name与df2的name相一致的所有行



#### ggplot()绘图

```R
ggplot(data = <DATA>) + 
   <GEOM_FUNCTION>(mapping = aes(<MAPPINGS>))
```

模板函数，

```R
ggplot(data = d) +
  geom_point(mapping = aes(x = year, y = carbon_emissions)) +
  xlab("Year") +
  ylab("Carbon emissions (metric tons)") +
  ggtitle("Annual global carbon emissions, 1880-2014")
```

例子

ggplot创建一个plot，数据选择为d

加号用来设定ggplot的属性。



#### 数据规整

##### pivot_longer()【将短表格转化为长表格】

类比于pandas pivot

```R
long <- plant_height %>%
  pivot_longer(
    cols = A:D,
    names_to = "plant",
    values_to = "height"
  )
```

##### pivot_wider()【将长表格转化为短表格】

```R
wide <- long %>% 
  pivot_wider(
  names_from = "plant",
  values_from = "height"
)
```

##### fill（）填充缺失

```R
sales %>% fill(year)
```

默认填充下方值，可以通过direction填充上方值

```R
sales %>% fill(year, .direction = "up")
```

##### drop_na()

去掉有空值的行

也可以指定某一列

drop_na(score)

##### replace_na()

将空值替代

replace_na(score,0)

