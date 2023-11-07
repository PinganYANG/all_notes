基础SQL语法

select product_id from Products **where** (low_fats **=** 'Y' and recyclable = 'Y')

select 选择 from 表 where条件



注意

​	null和其他值运算都为null

​	有与null相关的判断时要用is NULL



SQL server 利用 [] 来替代关键字

oracle PostgreSQL 和ANSISQL 用双引号

MySQL 用反引号``



排序 order by  columns ASC升序DESC降序



然后去重用SELECT DISTINCT



求字符串长度

​	除了SQL server都用length



Join

​	可以用 inner join / left join/ right join/ outer join 并且用法就是

​	SELECT * from table1 inner/left/right/outer join table2 on table1.id = table2.id



SQL中的subquery

1. **子表在FROM子句中**: 在这种情况下，子查询的结果被当作一个临时表，你可以对其进行进一步的查询操作。

   ```sql
   SELECT a.columnName
   FROM (
       SELECT columnName
       FROM yourTableName
       WHERE someCondition
   ) AS a
   WHERE a.columnName = someValue;
   ```

2. **子表在WHERE子句中**: 这里，子查询可以为主查询提供一个值或一组值。

   ```sql
   SELECT columnName
   FROM yourTableName
   WHERE yourColumn IN (
       SELECT anotherColumn
       FROM anotherTableName
       WHERE someCondition
   );
   ```

3. **子表在SELECT子句中**: 子查询可以在`SELECT`子句中使用，为每行返回一个值。

   ```sql
   SELECT columnName,
          (SELECT someAggregateFunction(anotherColumn)
           FROM anotherTableName
           WHERE yourTableName.someColumn = anotherTableName.someColumn) AS derivedColumn
   FROM yourTableName;
   ```

注意，在使用子表时，尤其是在`FROM`子句中，通常需要为其指定一个别名（如上述示例中的`a`）。此外，某些数据库系统可能有对子查询深度或数量的限制。



注意 from/inner join后的表明后可以接一个缩写字母，方便后面的on等操作



使用count可以进行计数，count，avg，sum等是和group by 一同使用的聚合函数

然后subquery可以放在from的括号后，后面加上as t等缩写

```sql
select customer_id,count(customer_id) as count_no_trans from (select customer_id,Transactions.visit_id from Transactions right join Visits on Transactions.visit_id = Visits.visit_id) as a where visit_id is NULL group by customer_id
```



可以用DATE_ADD,DATE_SUB (col, INTERVAL 1 DAY) 在MySQL中改变日期

用+ 1 INTERVAL ‘1 day’ 在PostgreSQL中改变日期



可以用内置函数datediff判断日期差



```sql
# Write your MySQL query statement below
select total.machine_id,ROUND(AVG(total.processtime),3) as processing_time
from 
(select a1.machine_id,(a2.timestamp - a1.timestamp) as processtime
  from Activity a1 inner join Activity a2
  on a1.machine_id = a2.machine_id 
    and a1.process_id = a2.process_id
    and a1.activity_type = 'start'
    and a2.activity_type = 'end'
) as total  
group by total.machine_id
```

用join时可以使用类似and a1.activity_type = 'start' 的筛选，并且可以在select中利用(a2.timestamp - a1.timestamp)来求一些值

记得用聚合函数AVG和内置函数ROUND

用IFNULL来将null替换为0



```sql
elect ss.student_id,ss.student_name,ss.subject_name,count(e.subject_name) as attended_exams  from (select s.student_id,s.student_name,su.subject_name from Students s cross join Subjects su) as ss left join  Examinations e  on e.student_id = ss.student_id and ss.subject_name= e.subject_name
group by ss.student_name,ss.subject_name
order by ss.student_id,ss.subject_name

```

注意，select出来的列要注意来自哪个子表

注意，如果不想要输出时，记得用Where is not null

```sql
SELECT score,
       CASE 
           WHEN score >= 90 THEN 'A'
           WHEN score >= 80 THEN 'B'
           WHEN score >= 70 THEN 'C'
           WHEN score >= 60 THEN 'D'
           ELSE 'F'
       END AS grade
FROM students;
```



可以用CASE END 根据条件加入新的列

时间条件 在start和end之间的判断可以用between



##### BETWEEN 

between 开始时间('1999-01-01') and 结束时间（'2333-01-02')

where条件要在group by前面



有两个排序条件时，可以用两个顺序的order by 

 order by  percentage DESC, r.contest_id ASC

可以通过 (select count(*) from Users) 强行得到行数



可以用 IF（条件，v1，v2）临时增加一列



```sql
DATE_FORMAT(trans_date, '%Y-%m')
```

来改变DATE的格式





```sql
select DATE_FORMAT(t.trans_date, '%Y-%m') as month,t.country,count(t.country) as trans_count,sum(IF(t.state = 'approved',1,0)) as approved_count,sum(t.amount) as trans_total_amount  ,sum(IF(t.state = 'approved',t.amount,0)) as approved_total_amount  from Transactions t
group by month,t.country
```

利用聚合函数可以实现很复杂的变换



count函数也会把0算进去，但不会吧NULL算进去



```sql
select round (
    sum(order_date = customer_pref_delivery_date) * 100 /
    count(*),
    2
) as immediate_percentage
from Delivery
where (customer_id, order_date) in (
    select customer_id, min(order_date)
    from delivery
    group by customer_id
)
```

这里用where 用了subquery



这里用where是保证了如果直接用where内的subquery会产生min后的值无法对应的问题

用类似上文的where subquery选择要使用in 表，=的话等号右侧只可以是单行



```sql
	SELECT person_name, @pre := @pre + weight AS weight
	FROM Queue, (SELECT @pre := 0) tmp
	ORDER BY turn

```

sql 中可以用@value 创建自变量，然后用 @value := 来为其赋值









```sql
SELECT 
    'High Salary' category,
    SUM(CASE WHEN income > 50000 THEN 1 ELSE 0 END) AS accounts_count
FROM 
    Accounts


```

可以用'High Salary' category 这种方法创建一个新列，列的行数和category行数相同，并为其赋值。此处的结果是一行，因为使用了SUM聚合函数使得结果仅有一行



利用Union可以合并不同的select出来的表，用来进行一些分类的查询

```sql
SELECT 
    'Low Salary' AS category,
    SUM(CASE WHEN income < 20000 THEN 1 ELSE 0 END) AS accounts_count
FROM 
    Accounts
    
UNION
SELECT  
    'Average Salary' category,
    SUM(CASE WHEN income >= 20000 AND income <= 50000 THEN 1 ELSE 0 END) 
    AS accounts_count
FROM 
    Accounts

UNION
SELECT 
    'High Salary' category,
    SUM(CASE WHEN income > 50000 THEN 1 ELSE 0 END) AS accounts_count
FROM 
    Accounts

```



**窗口函数**

这是一个使用SQL窗口函数（window function）的示例。窗口函数允许你在一个结果集的子集上执行计算，这些子集是基于当前的行确定的。

这行SQL具体做的是：

```sql
SUM(amount) OVER (ORDER BY visited_on RANGE BETWEEN INTERVAL '6' DAY PRECEDING AND CURRENT ROW) AS sum_amount
```

解释如下：

1. `SUM(amount)`: 这部分对`amount`列求和。

2. `OVER`: 该关键字表示我们要使用窗口函数。

3. `ORDER BY visited_on`: 这表示我们要按`visited_on`列排序数据。这样，对于每一行，SQL知道它在时间线上的位置。

4. `RANGE BETWEEN INTERVAL '6' DAY PRECEDING AND CURRENT ROW`: 这是窗口函数的核心部分。它定义了一个范围，告诉SQL如何为当前行确定子集。具体来说，对于当前行，它会取前6天的数据（包括当前行），然后对这些数据的`amount`列求和。

   - `INTERVAL '6' DAY PRECEDING`: 表示从当前行开始，向前数6天。
   - `AND CURRENT ROW`: 表示这个范围包括当前行。

5. `AS sum_amount`: 这部分为这个计算结果命名为`sum_amount`。

总之，这行SQL为每一行计算了前6天（包括当前行）的`amount`列的总和，并将结果命名为`sum_amount`。

这种窗口函数在金融、统计和其他需要基于时间序列数据的领域非常有用。在上述的场景中，它可能用于计算过去7天（包括今天）的累计销售额或其他度量。

```sql
SELECT
    SUM(insurance.TIV_2016) AS TIV_2016
FROM
    insurance
WHERE
    insurance.TIV_2015 IN
    (
      SELECT
        TIV_2015
      FROM
        insurance
      GROUP BY TIV_2015
      HAVING COUNT(*) > 1
    )
    AND CONCAT(LAT, LON) IN
    (
      SELECT
        CONCAT(LAT, LON)
      FROM
        insurance
      GROUP BY LAT , LON
      HAVING COUNT(*) = 1
    )
;

```

利用concat可以把两列合并在一起

并且利用 group by 和 Having count(*)的组合，可以实现去重和找唯一等条件



```sql
select e1.Name as 'Employee', e1.Salary
from Employee e1
where 3 >
(
    select count(distinct e2.Salary)
    from Employee e2
    where e2.Salary > e1.Salary
)
;
```

子查询的筛选：

​	首先子查询中的where确实是nxn的比较，但由于加上了一个count，就使得salary聚合为一行。 



SQL 提供了函数

​	**SUBSTRING**(col,idx_start,idx_end) 表示对col数据取idx_start 到 idx_end 的切片

​	**SUBSTRING**(col,idx_start) 表示对col数据取idx_start 到 最后 的切片

​	**UPPER** 取大写 **LOWER** 取小写



**LIKE 操作符**

SQL 的 `LIKE` 操作符用于在 `WHERE` 子句中基于指定的模式搜索列中的值。它通常与 `%`（代表零个、一个或多个字符）和 `_`（代表一个字符）这两个通配符一起使用。

以下是 `LIKE` 操作符的一些基本用法：

1. **% 通配符**:
   - 代表零个、一个或多个字符。
   
   示例:
   ```sql
   SELECT * FROM customers WHERE name LIKE 'Jo%'; 
   ```
   上述查询将会选择名字以 'Jo' 开头的所有客户。

2. **_ 通配符**:
   - 代表一个字符。
   
   示例:
   ```sql
   SELECT * FROM customers WHERE name LIKE 'Jo_n'; 
   ```
   上述查询将会选择名字为 'John', 'Joan' 等的客户，但不会选择 'Joon' 或 'Jon'。

3. **组合使用**:
   ```sql
   SELECT * FROM customers WHERE name LIKE 'J%n';
   ```
   上述查询将会选择名字以 'J' 开头，以 'n' 结尾的所有客户，如 'John', 'Jen', 'Jason' 等。

4. **不包含某个模式**:
   使用 `NOT LIKE` 来排除匹配某个模式的值。
   
   示例:
   ```sql
   SELECT * FROM customers WHERE name NOT LIKE 'Jo%';
   ```
   上述查询将会选择名字不是以 'Jo' 开头的所有客户。

使用 `LIKE` 时，需要注意数据库的大小写敏感性。某些数据库（如 MySQL 在某些配置下）默认不区分大小写，而其他数据库（如 PostgreSQL）默认区分大小写。如果需要进行大小写不敏感的搜索，可能需要使用特定于数据库的函数或操作。

总之，`LIKE` 是 SQL 查询中非常有用的工具，特别是当你需要基于部分匹配来搜索数据时。

##### DELETE 用法

DELETE from table where condition

##### UPDATE 用法

用法

```mysql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;
```

实际例子：

```mysql
UPDATE salary
SET
    sex = CASE sex
        WHEN 'm' THEN 'f'
        ELSE 'm'
    END;
```

这里用了case来条件设定值



##### DENSE_RANK() & RANK()

```sql
SELECT
    name,
    salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS SalaryRank
FROM
    employees;

```

denserank的实际作用和pandas的rank('dense')相似。OVER后接着的是rank的排序

这里可以换为RANK

##### ROW_NUMBER()

`ROW_NUMBER()` 是 SQL 中的一个窗口函数，用于为查询结果中的每一行分配一个唯一的序号。这个序号是连续的整数，通常从1开始，并根据指定的顺序依次递增。`ROW_NUMBER()` 在处理数据分页、排名和数据切分等场景中非常有用。

`ROW_NUMBER()` 函数的基本语法如下：

```sql
ROW_NUMBER() OVER (ORDER BY column_name [ASC|DESC])
```

- `OVER` 子句用于定义窗口，指定行号的分配是基于哪些列和顺序。
- `ORDER BY` 指定了排序的列和顺序（升序或降序）。

这样相当于加一个顺序int id



```sql
SELECT
    E1.id,
    E1.month,
    (IFNULL(E1.salary, 0) + IFNULL(E2.salary, 0) + IFNULL(E3.salary, 0)) AS Salary
FROM
    (SELECT
        id, MAX(month) AS month
    FROM
        Employee
    GROUP BY id
    HAVING COUNT(*) > 1) AS maxmonth
        LEFT JOIN
    Employee E1 ON (maxmonth.id = E1.id
        AND maxmonth.month > E1.month)
        LEFT JOIN
    Employee E2 ON (E2.id = E1.id
        AND E2.month = E1.month - 1)
        LEFT JOIN
    Employee E3 ON (E3.id = E1.id
        AND E3.month = E1.month - 2)
ORDER BY id ASC , month DESC
;

```

用left join 中的 `(E3.id = E1.id AND E3.month = E1.month - 2)`中的E1 month -2 来对应join不同行的值。【这里比pandas好用



##### cross join

纯m*n的叠加，没有on条件



##### left outer join & right outer join

虽然没有pandas中的outer join，但可以用left outer join 来实现

##### LIMIT

- LIMIT后接上单值，是类似head的效果
- LIMIT后接上两值，如`LIMIT M,1` 则是从第M行开始向下取1行



where 判断是否Null

- 判断是null：
  - where col **is** null
- 判断不是null:
  - where col **is not** null



#### 连续时间问题

```sql
with t1 as(
    select *,id - row_number() over(order by id) as rk
    from stadium
    where people >= 100
)
```

首先with是可以创造一个临时表



然后利用*.id - row_number()并给予Where的条件，就可以将连续的行分组。因为id是连续的，在用where筛选后就变为了断裂的，而row_number仍然是连续的，这样就会有差值。如图

![c99beb17bd069361beee0b78ea3eff1.png](https://pic.leetcode-cn.com/1617614624-EtXYFq-c99beb17bd069361beee0b78ea3eff1.png)

#### 函数

```mysql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
DECLARE M INT; 
    SET M = N-1; 
  RETURN (
      SELECT DISTINCT salary
      FROM Employee
      ORDER BY salary DESC
      LIMIT M, 1
  );
END
```

- Begin后的Declare可以声明一个用在函数内的变量，用set来赋值
- LIMIT可以巧妙出值
