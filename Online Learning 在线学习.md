#### Online Learning 在线学习

在线学习算法的特点是：每来一个训练样本，就用该样本产生的loss和梯度对模型迭代一次，一个一个数据地进行训练，因此可以处理大数据量训练和在线训练。

Online Learning 致力于**降低regret和提高sparsity**

$Regret=\sum^T_{t=1}{l_t(W_t)}-min_w\sum^T_{t=1}{l_t(W)}$

即**第t轮迭代后的累计损失和所有样本"*最优解*"的累计损失差值尽量小。**第二项$min_w\sum^T_{t=1}{l_t(W)}$表示所有样本累计损失和的最优解。

Sparsity即是稀疏性。传统的OGD方法一是由于浮点运算的原因即是加入了L1正则项也无法实现绝对的零；二是由于每次随机沿着一个样本的梯度方向更新，使得寻优过程更随机。



FTRL

![img](https://images0.cnblogs.com/i/417893/201406/262038582382250.png)

工程上可以采用FTRL方法

##### OGD Online Gradient Descent

在线学习不需要一次性访问所有的数据。相反，它会每次处理一个数据点，并相应地更新模型。

OGD 的一个关键特点是它每次只使用一个数据点来更新模型，这使得它能够快速适应新数据。但这也意味着，与传统的批量梯度下降相比，它可能更加容易受到噪声的影响。并且容易收到最新的数据点的噪声影响。

##### FOBOS Forward-Backward Splitting method 

基本思想：跟projected subgradient方法类似，不过将每一个数据的迭代过程，分解成一个经验损失梯度下降迭代和一个最优化问题。分解出的第二个最优化问题，有两项：第一项2范数那一项表示不能离第一步loss损失迭代结果太远，第二项是正则化项，用来限定模型复杂度抑制过拟合和做稀疏化等。

##### RDA **Regularized dual averaging**

　　　　–非梯度下降类方法，属于更加通用的一个primal-dual algorithmic schema的一个应用

　　　　–克服了SGD类方法所欠缺的exploiting problem structure，especially for problems with explicit regularization。

　　　　–能够更好地在精度和稀疏性之间做trade-off