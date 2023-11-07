#### LSTM

##### 介绍

时间序列预测任务。对方的公司Epalia是一家回收palette的公司。

projet的目的是预测各个point relais的palette收集数量。

##### 数据

数据就是全法国内各个point relais的两年内的palette收集数量。自变量是时间，因变量就是palette数量。

##### 难点

- 首先point relais有上百个，而且没有等级的区分，都属于同一种类。
- 不同的point relais的区别很大。有的point relais收集的数量很大，每个工作日都会收集palette。而有的point relais的收集数量低，并且每两次收集的间隔时间差别很大，从一个星期到一个月不等。后一种对于时间序列预测是很不好的。
- 这样的数据对于时间序列预测很不好

##### 亮点

- 因此进行了一些数据处理的方法：
  - 首先和epalia方面进行了沟通，去除了一些不重要的且数据波动大的point relais
  - 对于数据较好的point relais直接预测；对于数据具有波动但波动不是特别剧烈的小point relais采用了kmeans的聚类方法，对不同的cluster分别预测
- 聚类和filter虽然可以降低数据的波动，但波动仍然较大。直接使用LSTM模型还是会有较强的自相关性。即使进行了差分的方法，LSTM仍然趋向于将前一天的实际值作为后一天的预测值。
- 因此采用了EMD（empirical mode decomposition）的方法。分解原波动信号，以削弱其波动性，然后对每个信号分别进行时间序列预测，最后合并。这样可以完全削弱其自相关性了。

##### 模型结构

数据处理----filtering/clustering----EMD----LSTM

##### 不足

- 这个模型预测后一天还可以，但预测后一周总量，后一月总量时，其总数据量就不足了，难以直接预测。

- 而要想连续预测5天的数据，即将前一天的预测输出作为后一天的预测输入，则差距较大。

#### CV

##### 介绍

利用Computer Vision的方法，识别无人机在高压电线塔绝缘子上的缺陷。

##### 数据

- 无人机的录像
- 无人机拍摄的照片

##### 难点

- 无人机拍摄的照片没有进行缩放，目标在整体中过小
- 目标背景千变万化，而且比较复杂。比如有森林，城市，土地等等。这样将目标完全分割出来就比较有难度了。

##### 亮点

- 利用了Segment Anything Model的pretrained模型，利用YOLO网络作为Prompt工程部分，即将YOLO识别出的boundingbox作为Prompt输入到SAM中，得到目标的分割。并加入了alpha matting模块，来精细化分割结果的边缘，从无人机拍摄的复杂背景之中分割出目标

- 利用了无监督的FastFlow缺陷识别方式来对分割出的目标进行缺陷识别，因为负样本本身较少，而且所在的位置和大小会发生变化。因此无监督的模型可以较好的适应。

  

##### 模型结构

![image-20231104101233152](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231104101233152.png)

- SAM是transformer结构
- FastFlow是ResNet作为feature extractor

##### 不足

- 首先不是一个End-to-End的网络，是各个网络的组合，不同的功能由不同的模块组成，而且运行时间较慢。但各个功能的模块化保证了稳健性和industrialisation中的可优化性。

#### Similarity

##### 介绍

Les produits de sons, genre musique et podcast

寻找不同产品的相似度

##### Challenge

Beaucoup des données et pas bien structuré



##### Solution

Après la nettoyage avec python, nous essayons deux méthodes pour trouver la similarité.

- une est density based, nous trouvons que DBSCAN est la meilleure. 

- L'autre est inspiré par spectral clustering. nous établissons une matrice de produit fois produit avec la quantité de client qui utilise ces deux produits en même temps. La distance entre deux produits est l'inverse de la quantité de client. 

##### 数据

- 非结构化的大量数据，产品的id和对应的大量信息

##### 难点

- 难点主要出现在数据处理方面，因为数据量很大，而且是非结构化的，比如有嵌套字典列表的情况，因此处理起来比较麻烦

##### 亮点

- 首先就是用了比较传统的kmeans以及DBSCAN方法来对产品进行聚类，发现DBSCAN效果更好，各个clustering有明显的性别、年龄区分
- 然后就是受到了spectual clustering的启发，创建了一个产品*产品的矩阵，values是同时使用两个产品的客户数量。这样每个产品都是一个小cluster，和其他产品的相似度由同时使用cluster产品和目标产品的客户数量。这样就可以进行产品推荐。

##### 结构

- kmeans or DBSCAN
- 自创的矩阵

##### 不足



#### LLM

##### 介绍

首先就是面对ChatGPT的爆火，以及ChatGPT可能出现的Data Leak问题，因此要尝试搭建一个基于开源LLM大模型的可以搭建在本地服务器的类ChatGPT的模型。并且要让他基于一个本地知识文档，让他可以回答一些企业内的知识问题，帮助企业和客户。

##### 数据

- 这个框架本身不需要数据，以开发为主

##### 难点

- 

##### 亮点

- 亮点就是实现了一个本地的可以更改后端模型的框架。可以根据一些需求更换开原模型。比如最近LLama2开源了并且可以商用，其效果就比较好。而且后端根据需求也可以自行选择是否用ChatGPT的API

##### 结构

![image-20231104104054678](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231104104054678.png)



##### 不足



#### LLM应用在网页ChatBot上

##### 介绍

在上述框架搭建好后，就接到了一个基于网页文档的ChatBot的需求。

这是一个关于高血压产品和医疗帮助的网站，目的是帮助网站主，一个教授，建立一个ChatBot来帮助教授解答一些问题，减少教授的沟通需求和压力。

##### 数据

数据本身就是网站内的一些知识文档

##### 难点

- 最重要的难点就是要生成准确的Prompt
  - 首先是医学方面的问题必须要保证回答的准确性
  - 其次是LLM模型的一个问题，如果Prompt不够好他就会随机生成一些真假难辨的回答，

##### 亮点

- 通过观察文档数据可以看出，文档的关键信息都在对应网站的url上。因此提出了一个模块，根据query筛选生成Prompt的文档来源，使生成Prompt的文档来源，即url都是与query的关键信息相对应的。

##### 结构

![image-20231104104054678](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231104104054678.png)

##### 不足

- 开源的模型效果并没有ChatGPT3.5好，最好是进行一些finetuning的工作来提升模型的表现。