#### 插值 interplotation

最普通的就是用plt.plot 画图

并且用plt.title设置图片title

用plt.xlabel 设置x轴名称

plt.ylabel 设置y轴名称

##### 插值的用处

填充缺失位置信息



##### 线性插值

用scipy.interpolate 的 `interpld` 默认进行线性插值

```python
interpld(x,y,kind='linear')
```



而interpld的kind 关键字 可以用来选择不同的插值方式：

- `nearest` 最近邻插值

- `zero` 0阶插值

- `linear` 线性插值

- `quadratic` 二次插值

- `cubic` 三次插值

- `4,5,6,7` 更高阶插值

  

##### 如何选择插值方式

根据数据特性，插值位置，平滑度，数据量等进行选择



##### 径向基函数

径向基函数，简单来说就是点 $x$  处的函数值只依赖于 $x$ 与某点 $c$ 的距离：
$$
\Phi(x,c) = \Phi(\|x-c\|)
$$

##### 径向基函数插值

用`from scipy.interpolate.rbf import Rbf`

`cp_rbf = Rbf(data['TK'], data['Cp'], function = "multiquadric")`

不同的核函数取自不同的function

##### 为什么要用径向基插值

- 处理散乱数据点，即数据点**不是均匀分布在网格的情况上**，常见于地球科学
- 光滑性
- 高位数据
- 可以一定程度上外推





#### 统计分析

##### 正态分布

`from scipy.stats import norm`

它包含四类常用的函数：

- `norm.cdf` 返回对应的[累计分布函数](https://zh.wikipedia.org/wiki/%E7%B4%AF%E7%A7%AF%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0)值【F】
- `norm.pdf` 返回对应的[概率密度函数](https://zh.wikipedia.org/wiki/%E6%A9%9F%E7%8E%87%E5%AF%86%E5%BA%A6%E5%87%BD%E6%95%B8)值【f】
- `norm.rvs` 产生指定参数的随机变量
- `norm.fit` 返回给定数据下，各参数的[最大似然估计](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1)（MLE）值

其中学生分布等用norm.pdf(array,degree_of_freedom)



##### 离散分布

`from scipy.stats import binom, poisson, randint`





##### 假设检验

可以实现各类假设检验



#### 曲线拟合

##### 多项式拟合

用numpy的polyfit函数即可

##### 最小二乘拟合线性

用

`scipy.linalg.lstsq`

##### 线性回归

`scipy.stats.linregress`

##### 最小二乘拟合非线性

用

`from scipy.optimize import leastsq`

`leastsq(loss_func,[init_values],args=[values in loss func without first one])`

注意：

- 输入时他会将参数展平
- 返回时返回的是一系列对应的预测值和实际的插值的list而不是sum



##### 直接用curve_fit拟合非线性

`from scipy.optimize import curve_fit`

`curve_fit(目标曲线, x, y_noisy)`

这里的curve_fit可以直接返回 **目标曲线函数** 中作为参数的部分

比如

```python
def model_func(x, a, b, c):
    return a * x**2 + b * x + c
```

就可以返回a,b,c三值

#### 最小化

`from scipy.optimize import minimize`

`minimize` 接受三个参数：第一个是要优化的函数，第二个是初始猜测值，第三个则是优化函数的附加参数，默认 `minimize` 将优化函数的第一个参数作为优化变量，所以第三个参数输入的附加参数从优化函数的第二个参数开始。

```python
def neg_dist(theta, v0):
    return -1 * dist(theta, v0)
```

`minimize(neg_dist, 40, args=(1,))`

默认使用BFGS方法

可以用method关键字改变使用的算法



#### 积分【没用scipy先不看了】

#### 解微分方程【好像也还用不到】

#### 稀疏矩阵

`Scipy` 提供了稀疏矩阵的支持（`scipy.sparse`）。

稀疏矩阵主要使用 位置 + 值 的方法来存储矩阵的非零元素，根据存储和使用方式的不同，有如下几种类型的稀疏矩阵：

类型|描述
---|----
`bsr_matrix(arg1[, shape, dtype, copy, blocksize])` | Block Sparse Row matrix
`coo_matrix(arg1[, shape, dtype, copy])`    | A sparse matrix in COOrdinate format.
`csc_matrix(arg1[, shape, dtype, copy])`    | Compressed Sparse Column matrix
`csr_matrix(arg1[, shape, dtype, copy])`    | Compressed Sparse Row matrix
`dia_matrix(arg1[, shape, dtype, copy])`    | Sparse matrix with DIAgonal storage
`dok_matrix(arg1[, shape, dtype, copy])`    | Dictionary Of Keys based sparse matrix.
`lil_matrix(arg1[, shape, dtype, copy])`    | Row-based linked list sparse matrix

在这些存储格式中：

- COO 格式在构建矩阵时比较高效
- CSC 和 CSR 格式在乘法计算时比较高效

#### 线性代数

##### np矩阵【一般不用】

`np.mat np.matrix`都可以生成np矩阵。并且这矩阵可以

- A.T 求转置
- A.I 求逆



默认用 `*` 表示矩阵乘法

##### np ndarry 【用的比np.matrix多】

`A = np.array([[1,2], [3,4]])`

- `np.linalg.inv(A)` 求逆【比matrix复杂一点】
- `A.T` 求转置
- `np.dot(M,N)`求矩阵乘法
- `A * b` 求对应元素相乘，即一列与一行对应元素相乘，但不相加
- `np.linalg.det(A)` 求行列式



##### np.linalg.norm

np.linalg.norm(A) 默认是L2模数,即 平方和求根号，欧式距离

np.linalg.norm(A,1) 默认是L1模数，绝对值的和

np.linalg.norm(A,0) 默认是0范数，不为0值的数量



##### 矩阵分解——特征值



- `linalg.eig(A)` 
    - 返回矩阵的特征值与特征向量
- `linalg.eigvals(A)`
    - 返回矩阵的特征值
- `linalg.eig(A, B)`
    - 求解 $\mathbf{Av} = \lambda\mathbf{Bv}$ 的问题

##### 矩阵分解——奇异值分解

$M \times N$ 矩阵 $\mathbf A$ 的奇异值分解为：
$$
\mathbf{A=U}\boldsymbol{\Sigma}\mathbf{V}^{H}
$$
其中 $\boldsymbol{\Sigma}, (M \times N)$ 只有对角线上的元素不为 0，$\mathbf U, (M \times M)$ 和 $\mathbf V, (N \times N)$ 为正交矩阵。

\- `U,s,Vh = linalg.svd(A)` 

​    \- 返回 $U$ 矩阵，奇异值 $s$，$V^H$ 矩阵

\- `Sig = linalg.diagsvd(s,M,N)`

​    \- 从奇异值恢复 $\boldsymbol{\Sigma}$ 矩阵

##### 矩阵分解——LU分解

$M \times N$ 矩阵 $\mathbf A$ 的 `LU` 分解为：
$$
\mathbf{A}=\mathbf{P}\,\mathbf{L}\,\mathbf{U}
$$
$\mathbf P$ 是 $M \times M$ 的单位矩阵的一个排列，$\mathbf L$ 是下三角阵，$\mathbf U$ 是上三角阵。 

可以使用 `linalg.lu` 进行 LU 分解的求解：

##### 矩阵分解——QR分解

$M×N$ 矩阵 $\mathbf A$ 的 `QR` 分解为：
$$
\mathbf{A=QR}
$$
$\mathbf R$ 为上三角形矩阵，$\mathbf Q$ 是正交矩阵。

维基链接：

https://en.wikipedia.org/wiki/QR_decomposition

可以用 `linalg.qr` 求解。

