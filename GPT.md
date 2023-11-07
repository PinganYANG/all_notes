#### GPT

Generative Pre-trained transformer

基于一个Transformer架构，先在大规模语料上进行无监督的预训练，然后再在有标注的有监督数据集上进行fine tuning的方式。



#### 网络结构

##### Transformer Decoder

首先他只使用了Transformer的Decoder部分，并且只保留了一个Mask Multi-head attention的部分

Mask的部分和Transformer中mask的一样，预测一个词时会将其和后面的其他词mask掉

<img src="https://pic3.zhimg.com/80/v2-3050466633b81cb3f48f9f66f01043d2_1440w.webp" alt="img" style="zoom:50%;" />

一共用了12层的Transformer decoder

#### 训练过程

人家的名字中都带有pre-trained了，就不能避免一个预训练过程

##### GPT-1 无监督的预训练

基于一个语言模型训练。给定一个无标签的序列$\mathcal U=\{u_1,u_2,...,u_n\}$,最大化似然值：
$$
L_1(\mathcal U) = \sum_i{logP(u_i|u_{i-k},...,u_{i-1};\boldsymbol{\Theta} )}
$$
解释一下公式中的各部分：

- \( $p(x_{t+k} | x_t, x_{t+1}, ..., x_{t+k-1}; \theta) $\): 这部分表示在给定之前的\( k \)个单词后，下一个单词\($ x_{t+k}$ \)出现的条件概率。这里的\( $\theta$ \)代表模型的参数。

- \( $\sum_{t=1}^{T-k}$ \): 这是一个求和，它遍历整个文本序列，从第一个单词到倒数第\( k \)个单词。

最大化这个似然值的意义是：

1. **模型准确性**: 通过最大化似然值，我们确保模型能够为给定的序列生成或预测尽可能高的概率。这意味着模型更有可能正确预测下一个单词。

2. **数据拟合**: 最大化似然值也意味着模型能够更好地拟合训练数据。这可以增加模型的泛化能力，使其在未见过的数据上表现得更好。

3. **直观理解**: 从直观上讲，最大化似然值意味着我们正在寻找一组参数，这组参数使得观测到的数据序列在模型中出现的概率最大。

【和猜的相同，就是单纯的最大化预测准确率】

然后输出就是：

$h_0 = UW_e+W_p$

$h_l = tranformer\_block(h_{l-1})\forall i \in [1,n]$

$P(u) = softmax(h_nW_e^T)$

这样就可以计算概率

##### GPT-1 有监督的微调

对于一组有标签$y$的token: $\{x^1,...,x^m\} $ ，现将这些token输入到预训练模型中，得到特征向量$h_l^m$，然后通过一个**全连接层**得到预测结果:
$$
P(y|x^1,...,x^m)=softmax(h_l^mW_y)
$$
其中$W_y$是全连接层的参数。那么有监督的目标就是**最大化**：
$$
L_2(\mathcal C) = \sum_{x,y}{P(y|x^1,...,x^m)}
$$
在论文中将这一目标函数增加了$L_1$的加权，最大化
$$
L_3(\mathcal C) = L_2(\mathcal C) +\lambda L_1(\mathcal C)
$$
λ一般取0.5。

微调时，指训练输出**全连接层**的$W_y$和**分隔符的嵌入值**【什么东西？？？？】

对于不同问题有不同处理方式：

![img](https://pic4.zhimg.com/80/v2-ec3f2132533559b7c054edbed946afbb_720w.webp)



##### GPT-1 训练细节

TODO

激活函数 GLEU【什么鬼



##### GPT-2

结构没什么创新，但主要思想是：

当一个语言模型的容量足够大时，它就足以覆盖所有的有监督任务，也就是说**所有的有监督学习都是无监督语言模型的一个子集**。例如当模型训练完“Micheal Jordan is the best basketball player in the history”语料的语言模型之后，便也学会了(question：“who is the best basketball player in the history ?”，answer:“Micheal Jordan”)的Q&A任务。【猴子打字机？？？？？？？？？这你妈的也行？？

##### GPT-2 训练细节

TODO



##### GPT-3 

海量数据和参数

##### GPT-3 训练细节

TODO

