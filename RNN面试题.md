#### 如何解决RNN梯度消失或梯度爆炸

对于梯度爆炸而言，利用gradient clipping方法，当梯度过大或过小时，直接截断

对于梯度消失而言，可以将sigmoid或tanh激活函数改为relu

#### RNN的经典结构

1对1 输入和输出一致，比如文本对应

1对多 图形生成文字，生成音乐等

多对多 机器翻译等

#### LSTM为什么存在两种激活函数sigmoid和tanh

​	因为sigmoid用于各个门上，用于判断信息记忆还是遗忘

​	而tanh作用于状态和输出上

#### LSTM结构

![img](https://pic3.zhimg.com/80/v2-ff1e53c5716da3fc54ed2578fec905f6_720w.webp)

三个门

输入 用sigmoid值决定更新长期记忆Ct中的哪些部分，并用它来处理由短期记忆和输入X决定的长期记忆更改部分

遗忘 看看对上一个Ct要遗忘什么

输出 输出给下一个短期记忆

#### GRU结构

![img](https://pic3.zhimg.com/80/v2-5b805241ab36e126c4b06b903f148ffa_720w.webp)

**更新门** 用xt和ht-1来计算需要更新ht-1的部分

**重置门** 判断过去多少信息需要被遗忘

用充值后的和更新后的得到记忆和输出





#### 样本不均衡怎么解决

首先如果样本是简单的线性可分的，那么其实影响不大。

否则

1） 可以进行欠采样（随机欠采样，ENN **Edited Nearest Neighbors** 对于数据集中的每一个实例，使用k-最近邻算法来找到它的k个最近邻。如果一个实例的大部分最近邻属于其他类别，那么这个实例将被删除。）或过采样方法（随机过采样）。

2） 可以进行data augmentation 

3） 可以在损失函数方面进行修改

​       可以用class weight 方法，为不同类别提供不同权重

​        OHEM（Online Hard Example Mining）方法OHEM 的方法是在每次迭代中，首先使用当前的模型对整个数据集进行前向传播，然后根据损失值选择一些“困难”的样本进行反向传播。这样，模型的更新将主要集中在那些模型当前处理得不好的样本上。

​       Focal loss 特别设计用于处理高度不平衡的分类问题 

​       $FL(p_t)=-\alpha_t(1-p_t)^\gamma log(p_t)$ 这样pt是被分为正分类的概率。当概率pt接近1时，即分类正确时，就会使得损失值接近0，而概率接近0时，即分类错误时，则几乎不影响损失值，整体而言，相当于增加了分类不准确样本在损失函数中的权重。而α则直接影响正负样本权重。即通过αt 可以抑制正负样本的数量失衡，通过 γ 可以控制简单/难区分样本数量失衡。

4） 模型方面可以采用对样本不均衡不敏感的模型，比如LR或决策树，集成学习树模型。

5） 直接将其转化为一个异常检测问题

#### LSTM如何解决梯度消失的问题/为什么比RNN好

LSTM增加了更多回传梯度的路径，只要一条路径没有梯度消失，那么梯度消失的问题就得到了改善。因为LSTM有进有出且当前的**cell** informaton是通过input gate控制之后**叠加**的，**RNN是叠乘**，因此LSTM可以防止梯度消失或者爆炸。



#### 还有哪些其它的解决梯度消失或梯度爆炸的方法？

- 梯度裁剪gradient clipping，当BP时的梯度小于某个阈值或大于某个阈值时 ，直接裁剪，防止太小的梯度累乘带来的梯度消失或太大的梯度累乘带来的梯度爆炸。
- 改变激活函数，例如减少使用sigmoid、tanh这类激活函数，改成使用Relu、LeakRelu等，参考 [算法面试问题二（激活函数相关）](https://zhuanlan.zhihu.com/p/354013996) 。
- 残差结构，类似于CEC的模块，跨层的连接结构能让梯度无损的进行后向传播。
- Batch Normalization，相当于对每一层的输入做了一个规范化，强行把这个输入拉回标准正态分布*N~(0,1)。*这样使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数的大变化，进而梯度变大，避免产生梯度消失问题。而且梯度变化大 意味着学习收敛速度快，加快模型的训练速度。

#### Batch norm 和 layer norm 的区别

1. **计算轴的区别**：
   - **Batch Normalization (BN)**：BN 是沿着小批量数据的维度（通常是第0维）进行正规化的。换句话说，它独立地计算每个特征的均值和方差，基于整个小批量数据。【同时考虑每个样本中不同feature
   - **Layer Normalization (LN)**：LN 是沿着特征的维度进行正规化的。对于每个数据点，它计算所有特征的均值和方差。【在同一个样本之中共同考虑不同feature
2. **统计特性**：
   - **Batch Normalization (BN)**：因为 BN 是基于小批量数据进行计算的，所以它的统计特性（均值和方差）会随着每个小批量的数据变化而变化。
   - **Layer Normalization (LN)**：LN 的统计特性是固定的，因为它是基于单个数据点的所有特征进行计算的。
3. **使用场景**：
   - **Batch Normalization (BN)**：BN 主要用于前馈神经网络和卷积神经网络中【可以增加效率 加快收敛速度。
   - **Layer Normalization (LN)**：LN 通常用于循环神经网络（RNN）和 Transformer 结构中，因为它不依赖于小批量的大小，这使得它在处理序列数据时特别有用。【即使是单个输出也可以做norm
4. **外部参数**：
   - 无论是 BN 还是 LN，都有可学习的缩放和偏移参数，用于进一步调整正规化后的输出。
5. **对小批量大小的依赖**：
   - **Batch Normalization (BN)**：BN 对小批量的大小非常敏感。太小的小批量可能导致均值和方差的不稳定估计。
   - **Layer Normalization (LN)**：LN 完全不依赖于小批量的大小，这使得它在小批量大小变化时更为稳定。

#### CNN RNN 调参

momemtum 动量

epoch

learning rate

weight init （Xavier init）

regularization

​	dropout L1 L2

gradient clipping 

early stopping



num of layers

num of nodes

num of filters



#### 为什么用非线性激活函数

如果不适用非线性函数，多层的layer就毫无用处了

#### **什么是梯度爆炸**

误差梯度是神经网络训练过程中计算的方向和数量，用于以正确的方向和合适的量更新网络权重。

在深层网络或循环神经网络中，误差梯度可在更新中累积，变成非常大的梯度，然后导致网络权重的大幅更新，并因此使网络变得不稳定。在极端情况下，权重的值变得非常大，以至于溢出，导致 NaN 值。

网络层之间的梯度（值大于 1.0）重复相乘导致的指数级增长会产生梯度爆炸。

#### **梯度爆炸会引起什么问题**

在深度多层感知机网络中，梯度爆炸会引起网络不稳定，最好的结果是无法从训练数据中学习，而最坏的结果是出现无法再更新的 NaN 权重值。

#### One-stage

直接预测物体的类别和边框位置，比如YOLO【YOLO预测（c,x,y,w,h)】和SSD。

#### Two-stage

生成预选框，然后在预选框中预测类别和偏移量，比如RCNN家族

#### One-stage vs. Two-stage

- one-stage 速度快，精度低 vs two-stage 速度慢，精度高
- one-stage 实时响应 vs two-stage 准确性更高的

#### 交叉熵损失

$$
\begin{equation}
CE(p,y) = 
\begin{cases} 
-log(p) & \text{if } y = 1 \\
-log(1-p) & \text{otherwise }  
\end{cases}
\end{equation}
$$

$CE(p,y) = CE(p_t) = -log(p_t) = -ylog(p) - (1-y)log(1-p)$
$$
\begin{equation}
p_t = 
\begin{cases} 
p & \text{if } y = 1 \\
1-p & \text{otherwise }  
\end{cases}
\end{equation}
$$






#### Focal Loss

One-stage方法精度低的一个原因就是正负样本**极度不平衡**



【

**极度不平衡**：

1. 目标检测算法为了定位目标会生成大量的anchor box
2. 而一幅图中目标(正样本)个数很少，大量的anchor box处于背景区域(负样本)，这就导致了正负样本极不平衡

】



来源于**交叉熵损失**

那么怎么由交叉熵损失改进为Focal loss的？

首先我们的目标是处理正负样本**极度不平衡**，所以第一步：

- 为正样本增加权重α
  - $CE(p,y)  = -\alpha ylog(p) - (1-\alpha)(1-y)log(1-p)$
  - 这个方法虽然增加了正样本权重，但无法控制正样本中，**易分类样本**和**难分类样本**的权重
- 增加了针对难易样本的调制系数$(1-p_t)^\gamma$
  - 当一个样本被分错（**难样本**）的时候，此时$p_t$非常小，那么调制系数$(1-p_t)^\gamma$接近 1，损失基本不会被影响；
  - 当遇到一个**易样本**的时候， $p_t\rightarrow 1$，因此有 $(1-p_t)^\gamma\rightarrow 1$ ，那么对于比较容易的样本，loss 就会降低，相当于权重降低；

最终结果为：

$FL(p_t) = -\alpha_t(1-p_t)^\gamma log(p_t) $







#### 项目问题

##### LSTM

请你描述一下数据

为什么选择LSTM模型

【LSTM相关问题

在您的项目描述中，您提到了一系列复杂的数据处理和时间序列预测任务。作为面试官，以下是我可能会问的一些问题：

###### 数据处理和理解

1. 您能详细说明在与Epalia沟通后决定去除的“不重要”点的标准是什么吗？

​	Nous avons étudié avec eux. Et nous décidons de ne pas considérer points de relais avec moins de 5 collections de palettes chaque mois.

1. 您是如何决定哪些point relais的数据“较好”，可以直接用于预测的？

​	Les point relais sont bien quand il y a des collection des palettes environs tous les jours.

1. 您提到使用k-means聚类对点进行分组，聚类的具体参数和决定因素是什么？

   Pour le méthode de clustering, il faut d'abord dire pourquoi nous voulons l'utiliser. (Si déjà parler des méthodes de traitement des données, sinon, parle en) Après ce genre de traitement des données, même si l'on a déjà supprimer les points peu importants, il y a des points de relais qui ont genre 10 collection chaque mois avec l'intervalle très varié entre deux collection. Et dans le cas et avec le but de prédire, les données comme cela ne permet jamais de bien prédire pour chaque point comme cela. Si vous voulez le faire, il faut d'abord faire une classification binaire pour décider s'il y aura une collection un jour, et après faire la prédiction. Mais avec les données on a, il n'y a pas de possibilité de faire ce genre de classification. Donc pour faire la prédiction, nous considérons de faire le clustering pour que nous puissons considérons plusieurs points de relais en même temps. Le manager de Epalia pense que notre méthode a raison.  

​	Le clustering est fait basé sur les caractéristiques géographiques. Nous trouver les longitudes et les latitudes pour chaque points relais. Et le k-means est fait avec la eulidean distance. Et la quantité sont décidé par le elbow method.

1. 请解释一下您在预处理数据时使用的具体方法，如您是如何进行差分的？

​	Pour la différence, nous utilisons la méthode de différence du premier ordre. C'est d'utiliser la différence entre le jour suivant et ce jour pour entrainer.

​	Et nous avons utilisé une méthode de EMD avant la différence. 

###### 模型选择与应用

5. 您选择使用LSTM进行时间序列预测的原因是什么？

​	Pour la prédiction de série temporelle, LSTM peut faire la prédiction en considérant les information long terme. C'est une bonne et classique méthode de prédire.

5. 您是否比较了LSTM与其他时间序列预测模型，如ARIMA、Prophet或RNN的性能？

Oui, nous avons testé ARIMA, mais le résultat est très mal pour nos données qui vaient beaucoup. Et pour RNN, la performance n'est pas assez bien mais RNN est plus vite.

5. EMD方法是如何帮助减少数据波动性的？您是如何验证它的有效性的？

EMD est empirical Mode Decomposition. Il peut décomposer une signale aux différent Intrinsic Mode Function. Et chaque fonction contient une partie de variance et tendance de signale originale. 

Et nous faison EMD pour les données originales, ensuite nous faison la prédiction pour chaque composition. Enfin, nous les combiner pour obtenir le résultat final. 
Avec EMD, les résultats toujours mieux dans notre cas. 

5. 您在实际应用中如何处理LSTM模型的参数调整和优化？

Les hyperparamètres de LSTM à améliorer sont learning rate, nb de LSTM layers et nb de nodes dans un LSTM layer.

Premièrement, nous avons utilisé le grid search pour chercher les hyperparamètre. Mais cela prend trop de temps. Donc nous avons testé un Baysian Paramètre séléction pour trouver la meilleure. 

###### 模型性能与评估

9. 您如何评估预测模型的性能？使用了哪些具体的指标？

Je me souviens que nous utilisons MSE.



9. 对于较长时间范围的预测不足，您有考虑过使用什么方法来改进吗？

Nous pensons que pour la période longue, il manque des données à entrainer un bon modèle.

 

9. 在模型评估时，您是否进行了交叉验证或使用了一个独立的测试集？

Oui, nous prendrons deux mois récents comme le test set.

###### 实际应用与后续优化

12. 模型在生产环境中的实际应用情况如何？

Nous avons pas déployé

12. 您在项目中遇到的最大技术挑战是什么，以及您是如何解决的？

Un c'est le pb de données, ils ne sont pas parfait. Nous avons travaillé avec Epalia pour supprimer les outliers. Et nous faisait un clustering pour merger des données pour bien prédire.

L'autre est le modèle de prédiction. Pour le modèle de prédiction genre LSTM, la performance est pas mal pour la prédiction d'un jour suivant. Mais si la prédiction est pour 5 jour, la performance n'est pas très bien. Pour le moment je pense que peut etre nous pouvons essayer le transformer pour la prédiction.

12. 如果要改进模型以便它能更好地进行长期预测，您会怎么做？

Peut etre changer la base à transformer.

###### 系统设计与实现

15. 您是如何处理LSTM的实时更新问题的，特别是在数据每天都在变化的情况下？

L'entraînement de notre modèle ne prend que 15 min je me souviens. Donc cela sera un online learning. Et le window d'entraînement peut être 2 année

15. 您在设计和部署这个预测系统时，考虑了哪些可扩展性和维护性问题？

Quand nous avons fini le projet, nous ne savons pas la tech de docker et le click. La seule considération est séparer les fonction en class self fonction. Mais pour le moment, je connais bien le docker, et je crois c'est beaucoup plus mieux de le déployer dedans un docker. 

###### 项目管理与团队协作

17. 您在这个项目中的角色是什么？

Nous sommes une équipe de Trois. Et moi c'est le head de tech. Je proposer cet algo et le réaliser. 

17. 您是如何与其他团队成员协作的，尤其是在数据选择和预处理阶段？

Nous avons discuté ensemble et avec l'entreprise pour décider comme supprimer les outliers, si clustering est pertinent, etc. La solution sont faite par équipe.

###### 项目成果与商业价值

19. 您能否说明该模型为Epalia公司带来的具体商业价值？

Je ne  sais pas si notre projet a bien inspiré epalia. Mais notre prof aime bien notre projet.

19. 模型是否有助于Epalia改善其物流和库存管理？

Pour moi, je pense que oui. 

- Premièrement, epalia peut voir la tendance de collection des palettes pour réagir.
- Deuxièmement, epalia peut créer des entrepot avec notre résultat de clustering.

###### 个人发展与学习

21. 在完成这个项目的过程中，您最大的学习收获是什么？

La pratique de data cleaning et processing pour les données de série de temps. Et la pratique de LSTM en utilisant un EMD pour l'améliorer.

21. 有没有在这个项目中您觉得特别自豪的成就？

Quand LSTM peut prédire un résultat pas mal même seulement pour un jour suivant, je suis très heureux.

###### 对未来技术的看法

23. 您认为未来在时间序列预测方面有哪些技术是值得关注的？

Risque de climat, risque de supply chain, et les sujets traditionnels, genre le stock, inventory etc. 

23. 如果有更多的数据或资源，您认为可以采用哪些方法来进一步改进预测的准确性？

这些问题旨在深入了解您的技术能力、解决问题的方法、以及在项目管理和团队合作方面的经验。准备答案时，应尽量提供具体的实例和数据来支持您的观点。

##### CV

描述一下数据

为什么不直接用CNN

请描述一下SAM模型

请描述一下FastFlow模型

SAM模型有什么优势

FastFlow模型有什么优势

作为面试官，我可能会从技术细节到项目管理等不同方面来提问。以下是一系列可能的问题：

###### 技术细节

1. 请解释一下Segment Anything Model (SAM)的工作原理以及如何训练的。

   Segment Anything Model est un modèle pretrained sur beaucoup des données. Il est basé sur transformer et il se compose de trois parties. 

   Image encoder pour comprendre les image, prompt encoder pour comprendre les prompts et un mask decoder pour générer un masque prédite.

   Nous avons vu qu'il est assez bien pour notre tache. Mais nous avons vu la possibilité de fine tuning avec les données. Avec Alpha Matting, nous pouvons générer une meilleure masque et nous pouvons les utilisons pour le fine tuning pour notre tache. 

2. 您提到使用YOLO网络作为Prompt工程的一部分，能详细说明这是如何工作的吗？

   (Si déjà parlé de SAM, dire directement, sinon, présenter le SAM) Comme nous avons dis, nous pouvons transformerons des prompts, les points, les bbox ou les textes, à un modèle de SAM. Et avec ces prompt, SAM peut segmenter l'object avec ce prompt. Avec yolo, nous pouvons générer les bounding boxes pour les isolateurs, et nous pouvons transférer ces bounding boxes comme les prompt à SAM. Et SAM peut générer une bonne segmentation. 

3. 能否详细说明alpha matting模块的作用以及它如何帮助改善分割结果？

​	alpha matting a le but d'extraire un objet de premier plan aux contours doux à partir d'une image d'arrière plan. 

​	$I = \alpha F + (1-\alpha )B$

​	

​	En effet, les résultats de SAM ne sont pas parfaits, quand il y a des backgrounds très variés et compliqués. Il reste de partie de background ou il manque d'une partie d'objet. Donc il faut utiliser alpha matting pour améliorer ces résultats. 

​	Mais pour moi, je pense que module alpha matting est seulement une module temporaire, nous pouvons faire la fine tuning de SAM pour ne pas utiliser cette module et accélérer la performance.



1. 请描述一下FastFlow缺陷识别模型的工作机制和它是如何实现无监督学习的。

FastFlow peut apprendre un mapping de flow depuis les features d'un image.  Et il peut apprendre et construire le mapping des features depuis des isolateurs normales. Et quand il y a une isolateur mal, le mapping de cette isolateur ne sont pas identique que le mapping des isolateurs normales. Avec cela, nous pouvons réaliser un unsupervised learning. 



1. 对于FastFlow模型，您是如何确定使用ResNet作为特征提取器的？有比较过其他的网络结构吗？

​	Nous avons vu l'article de FastFlow, il propose d'utiliser ResNet comme une backbone de structure CNN. Et il a aussi proposé d'utiliser Transformer avec cadre de CaiT et DeiT. Donc nous avons testé Backbone de Wide ResNet, ResNet18, CaiT et DeiT. Nous avons vu que le cadre de Transformer a toujours le pb de ralentissement, et la performance n'est pas assez bien que ResNet. Et peut être parce que Transformer peut se souvenir l'information de long terme, il y a toujours des noisies dans le background. Nous avons aussi testé VGG-16, VGG-19, ils ont presque la même performance mais leur temps de tourner sont plus long. Donc nous avons choisir le Wide ResNet.

1. 在项目中处理小目标检测问题时，您采取了哪些特别的技术或策略？

   Pour notre projet, nous avons utliser YOLO pour détecter les petit objects dans les images avec grande résolution. YOLOv5 a déjà bien développé pour détecter les petits objets. 

   Avec YOLO et SAM, nous pouvons bien créer des isolateur à détecter.

    

2. 您是如何处理不同背景下的目标分割的？有使用特别的数据增强或迁移学习技术吗？

​	Nous avons testé que pour la partie de Segment Anything Model, il peut réaliser de segmenter presque tous les isolateurs depuis tous les différent background. Le pb est pour les background très compliqué, le résultat de SAM va avoir des choses de background ou perdre une partie d'objet.

​	Pour résoudre ce pb, nous avons deux solutions. 

- Première est d'utiliser Alpha Matting 

- Deuxième est de fine tuning SAM avec les résultats de Alpha Matting comme Training Data



​	Dans notre projet, nous avons utilisé  la première partie, et la deuxième méthode n'est pas encore réalisé quand j'ai fini mon stage. En conclusion, nous avons avons l'idée de fine tuning.

1. 您在这个项目中使用的最大的无人机拍摄图像分辨率是多少？

   Il a la résolution de 8k, 8192 fois 5120.

###### 数据处理

9. 无人机拍摄数据的质量如何？您是如何解决可能的质量问题的？

​	La résolution est bien. Le problème est que le background sont toujours différent et variés. Et ceci est résolu par notre algorithme en utilisant yolo et SAM.

​	 

9. 您如何确保拍摄的图片覆盖了所有必要的角度和细节，用于缺陷检测？

   Pour les isolateurs, ils sont de forme de colonne. Donc avec une bonne condition de lumière et angle de caméra, nous pouvons prendre toutes les détails. La bonne angles sont 90 degrés.

10. 数据集是否平衡？如果不平衡，您如何处理类不平衡问题？

​	Non ils ne sont pas bien équilibre. Nous avons utilisé la méthode de Data Augmentation pour les samples négatives. Nous utilisons la même méthode de Data Augmentation que celle dans un autre article.  数据增强方法

如何保证Data Augmentation的可靠性？

###### 结果评估

12. 您如何评估您的模型的性能？使用了哪些指标？

Avec Image AUROC, image F1Score, Pixel AUROC et pixel F1Score. 

12. 您能分享一些关于模型在真实世界数据上的性能结果吗？

Je me souviens que le pixel auroc est plus de 0.98.

12. 在部署这些模型时，您是如何衡量它们的实用性和准确性的？

###### 系统设计

15. 您如何在模块化和系统整体性能之间找到平衡？

​	Pour le moment, il n'est pas une framework end-to-end. Donc la meilleure la performance de chaque module, la meilleure la performance de système total. Donc c'est un trade-off entre les performance de chaque module entre la vitesse de système. Mais nous avons vu que nous avons pas besoin de trouver les anomalies en temps réel, donc nous allons améliorer toutes les performances des module sans considérer la vitesse d'inférence de système total. 



15. 鉴于运行时间较慢的问题，有没有可能的优化方法或已经在考虑的解决方案？

​	Nous avons vu la possibilité de fine tuning de SAM. Et après la fine tuning de SAM, si le résultat est assez bon, nous pouvons supprimer la module de Alpha Matting pour accélérer notre méthode.

15. 如何处理实时应用中的延迟问题？比如，在在线监测中。

    Pas besoin.

###### 项目管理和团队合作

18. 您是如何在团队中分配和协调任务的

18. 请问您在项目中扮演了什么角色？主要负责哪些部分？

Je m'occupe de créer cet algorithme et le pipeline. Et j'ai aussi décidé comment nous pouvons l'améliorer. 

18. 在项目中遇到的最大挑战是什么？您是如何克服的？

La permière version avec SAM mais sans la module Alpha Matting. Nous ne savons pas comment nous pouvons régler le pb de mal segmentation de SAM. Si nous voulons fine tuning, il faut trop de temps de créer une base de données parce que le contour précis d'une isolateur est très compliqué. Il n'est pas possible de le construire de zéro. 

J'ai vu les articles et j'ai trouver que nous pouvons utiliser Alpha Matting pour résoudre ce pb. Et alpha matting peut aider de construire une base de données très bien pour le fine tuning de SAM. 

###### 未来展望

21. 如果有更多的资源（时间、资金、人员），您会如何改进这个项目？

Il faut plus des données des échantillons négatives pour ne pas utiliser la Data Augmentation. Et il faut aussi de créer une base de données pour fine tuning le SAM.

21. 您在这个项目中学到了哪些教训，可以应用到未来的项目中？



21. 对于目前的不足之处，您有什么计划或想法来解决它们？



###### 其他问题

24. 在工程实践中，您是如何确保模型的泛化能力的？
25. 您是否考虑将这个系统与其他类型的检测系统（如热成像）结合使用？
26. 您是否考虑了在极端天气条件下无人机和模型的性能？

这些问题可以帮助面试官更深入地了解您的项目经验、技术能力和问题解决能力。准备回答这些问题时，不仅要清楚地说明您的方法和决策过程，还应该准备一些具体的例子和结果来支持您的论点。

##### Similarity

在您的项目描述中，您提到了使用机器学习技术来解决音频产品相似性识别的问题。以下是作为面试官，我可能会提出的一些问题：

###### 数据处理

1. 您使用的Python库和工具是什么？为什么选择这些工具？

   Nous avons utilisé pandas et numpy. C'est vite, pratique et pertinent. Et il y a pas mal de manuel sur internet.

2. 在数据清理过程中，有哪些特别的挑战？您是如何克服这些挑战的？

Même si les données sont structurées. Nous ne connais pas sont type, c'est un file .idomaar. Et en effet il ne sont pas totalement structuré. Il y a des colonnes qui a des valeurs comme dicts mais sous le format de string. Et il y a aussi des noms avec les marques. Donc il est un peu difficile de extraire toutes les informations depuis ce fichier. 
Il est fait petit à petit, étape par étape. Première tirer toutes les données bien stucturées. Ensuite essayer de comprendre la structure des données mal structurée et essayer de créer des fonction pour tirer les info dedans.

###### 模型选择与应用

4. 您选择DBSCAN进行聚类的原因是什么？您是如何确定其参数（如邻域大小ε和最小样本数MinPts）的？

Parce que DBSCAN basé sur la densité. Cela est pertinent car nous voulons que dans notre cluster, chaque produit neighbors peut avoir une distance pertinente. La méthode de KMeans peut introduit un pb genre pour un produit très 'outlier', il peut etre lié à un cluter grand. Mais il est vraiement pas le même genre que les autres produits.

Les params sont déterminées par un grid search.

4. 相比于其他聚类方法，DBSCAN在您的应用场景中表现出了哪些优势？

 La méthode de KMeans peut introduit un pb genre pour un client très 'outlier', il peut etre lié à un cluter grand. Mais il est vraiement pas le même genre que les autres clients.

4. 您是否尝试过其他基于密度的聚类算法，比如OPTICS？

non

4. 您如何理解和应用谱聚类的原理来建立产品的相似度矩阵？

Spectual clustering utiliser directement la similarité entre les produit. Donc avec spectual clustering, nous pouvons créer les clusters où les produits dedans ont une similarité grande. Et nous chisissons la quantité d'utlisateur comme la mesure de similarité. 

###### 结果分析

8. 您如何衡量聚类的效果？使用了哪些指标？

Les produits créent les valeurs seulement quand il y a des utilisateurs les utiliser. Donc nous essayons de voir les caractéristiques d'utilisateurs des produits pour évaluer. Si les utilisateur a une caractéristique similaire, nous pensons que le clustering est assez bon.  

8. 通过您的相似度矩阵，您能分享一个具体的案例，说明如何根据这个矩阵推荐相似产品吗？

Si on a une utilisateur qui utilise produit 1 et produit 2, produit 3, où produit 1 2 3 ont une similarité grande, et en même temps, il y a un autre utilisateur utilise produit 1, nous pouvons le proposer produit 2 et 3. 

8. 您的方法在哪些方面可能会受限？有哪些潜在的改进空间？

###### 技术细节

11. 您如何处理大规模数据集？是否使用了分布式计算或云服务？
12. 在建立产品*产品矩阵时，有没有遇到内存不足的问题？如果有，您是如何解决的？

###### 项目管理

13. 在您的项目中，团队是如何分工的？您具体负责哪些部分？

Je m'occupe la partie de data cleaning et processing.

13. 您如何确保项目的进度和质量？使用了哪些项目管理工具？

###### 实际应用与业务影响

15. 您的相似度矩阵和聚类结果如何被业务团队利用？有哪些实际的应用案例？
16. 这个项目对业务或产品策略有哪些具体的影响？

###### 个人贡献与成长

17. 在这个项目中，您最自豪的技术成就是什么？
18. 您在项目中遇到的最大挑战是什么，您是如何克服的？

###### 对未来技术的看法

19. 您认为在相似度分析领域，将来有哪些技术可能会发展起来？
20. 如果您有机会改进这个项目，您会考虑使用哪些新技术或方法？

准备这些问题的答案时，记得提供具体的例子和经验，这样可以更好地向面试官展示您的技术能力和项目经验。

##### LLM

在您的项目描述中，您讨论了创建一个本地服务器上的类ChatGPT模型，并应用于网页ChatBot的过程。以下是我作为面试官可能会提出的问题：

###### 关于本地LLM框架

1. 您是如何选择开源大模型的？有哪些因素考虑在内？

Premièrement, il faut des llm open source. Ensuite, c'est la performance. Et pour la performance, c'est une trade off entre la performance et la vitesse d'inférence.

1. 本地部署过程中，您是如何处理和保证模型性能的？

Premièrement, selon le besoin, il faut choisir la paramètre de LLM. Il ne faut pas trop grand le LLM, il prendre trop de source à lancer. Et deuxièmement, il faut une server avec assez de source de GPU pour inférer.



1. 您如何确保本地模型的数据安全和隐私？

Nous utilisont une gestion par clé SSH pour la sécuriter.  

1. 在部署过程中，您遇到了哪些技术挑战？如何解决的？



1. 模型本地化会带来哪些额外的维护和运营需求？

Le coût de serveur et de source. Et il faut développer pour le bien surveiller.

1. 您如何在不使用ChatGPT API的情况下，保持或提高模型的响应能力和准确性？

La présition des réponses sont garantie par le score sur MMLU. Et la vitesse de réponse est garentie par la source de serveur et l'export de llm si nous pouvons le faire.

1. 您是否进行了任何模型优化或定制开发以适应企业需求？

En effet oui. En raison de caractéristique de notre client, dans le domain médecine, il faut la précision de réponse. Donc j'ai utiliser le prompt engineering pour que le chatbot va répondre les question seulement basé sur notre prompt, ne pas créer ou tirer de son propre base de données. S'il n'y a pas de réponse pertinente dans le prompt, il ne réponse rien.

###### 关于网页ChatBot应用

8. 您如何确保医疗领域回答的准确性和合规性？

Nous utilisons un prompt engineering. Nous exigons llm de répondre seulement les info dans notre prompt. Donc si notre prompt est bon, la réponse doit etre bon ou pas de réponse. Cela suffit. 

8. 生成准确Prompt的过程中，您采取了哪些技术手段或算法？

Dans ce modèle, ce que nous faire est de créer un prompt basé sur le site. 

Premièrement, nous scraper les données depuis le site et créer une base de données de vecteur. Et pour chaque query, nous calculons la similarité entre le query et la base de vecteur. 

Deuxièmement, pour le query, nous faisons une preprocessing de tirer les mots clé et faire la filtre. Cela veut dire que nous allons créer les prompts avec la plus de similarité et avec la filtre de mots clés.

Et il faut ajouter la partie de constraint genre 'il faut répondre seulement basé sur le prompt, sinon, dit je ne sais pas'

8. 您是如何训练或准备模型来理解和生成医学领域的内容的？

Pour moi, j'ai pas le temps et j'ai pas assez de ressource de fine tuning un LLM. Mais moi, j'utilise prompt engineering. Je donne les prompts concernant la contenu de médecine. Cela exigons llm de répondre seulement les info dans notre prompt. Donc si notre prompt est bon, la réponse doit etre bon ou pas de réponse.



8. 您如何测试和验证ChatBot的性能和回答质量？

La performance est définie en deux partie:

- La vitesse, ce qui est évidant
- La qualité de réponse, ce qui est vu par un prof.

8. 在这个项目中，遇到最大的挑战是什么？您是如何应对的？



###### 技术和数据处理

13. 您在项目中使用了哪些NLP技术？

Embedding, calculation de similarité, inférence pour un retrival task 

13. 您是如何处理和整理网站内的知识文档的？

Je scraper le site avec langchain en téléchargant le url des site pour chaque doc. Ensuite j'ai utilisé un Embedding model pour établir une base de données de vecteur de Quadrant.

13. 在没有大量标注数据的情况下，如何进行有效的模型训练？

    

###### 模型训练和调优

16. 您是如何进行fine-tuning的？使用了哪些数据和技术？

J'ai pas fait fine tuning. Mais mon collegue a fait pour une tache de générer SQL depuis un texte. Je crois qu'ils congèlent les poids de layers et la régularisation. Et il utilise aussi la parallalisation sur GPU.  

16. 如果有限的fine-tuning数据，您是如何最大化模型性能的？
17. 您有没有考虑用迁移学习来提升模型在特定领域的表现？



###### 商业和实际影响

19. 这个项目对企业或客户的具体影响有哪些？

Ie client a déployé le chatbot sur leur site. 

19. 您如何衡量和跟踪ChatBot的成功？

###### 项目管理和团队合作

21. 在这个项目中，您的角色是什么？主要负责哪些方面？
22. 您是如何协调团队成员和跨部门合作的？
23. 您是如何在项目进度和预算内控制和交付结果的？

###### 对未来的展望

24. 您如何看待未来LLM在企业内部知识管理和客户服务中的应用？
25. 如果有机会，您会如何改进现有的系统？

在准备答案时，考虑提供具体的例子和经历，这会帮助面试官更好地理解您的技术能力和项目经验。



ETL, Extract Transform, Load



Pour cette partie, je m'occupe de construire et de maintenir un pipeline de a à z pour soutiens une application de pricing chez client. Le pipeline contiens de télécharger des données de côté de client et les cleaning, les processing avec un modèle de pricing de l'entreprise. Ensuite les uploader sur la base de données de Postgresql. Et j'ai aussi fait l'industrialisation d'utiliser le Airflow pour l'automatiser.

Le pipeline est fait avec python, je m'occupe aussi de créer les api pour communiquer les données entre python et SQL, le backend et le frondend. Et j'ai créé un pipeline de change data capture en temps réel totalement basé sur PostgreSQL avec la méthode de logical replication. Et une partie de mon pipeline est apprécié par mon manager, donc je l'ai ajouté dans le package interne de l'entreprise. 





1. **关于项目定价 (Pricing Project):**
   - 请详细介绍一下你负责的定价项目的具体内容，你在这个项目中扮演了什么角色？
   - 在构建和维护定价模型的过程中，你遇到了哪些挑战，又是如何克服这些挑战的？
2. **关于数据管道 (Data Pipeline):**
   - 请描述一下你创建的ETL数据管道的架构，以及你如何使用Amazon S3和Apache Airflow来实现这个管道。
   - 你如何确保数据管道的效率和数据的质量？
3. **关于API开发:**
   - 你可以详细说说你如何使用SQLAlchemy和FastAPI来构建前后端之间的API吗？
   - 在这个过程中，你是如何处理安全性问题，例如身份验证和授权的？
4. **关于CDC Pipeline (Change Data Capture):**
   - 你能详细解释一下你建立的CDC管道在PostgreSQL数据库中的作用吗？以及你是如何实现的？
   - 这个CDC管道是如何帮助你跟踪数据库变化的，你是如何解决可能出现的复制延迟问题的？
5. **关于Toolbox Packages改进:**
   - 请介绍一下你在gitlab上改进公司Toolbox packages的经历，你主要做了哪些改进？
   - 在改进过程中，你如何确保代码的可重用性和可维护性？
6. **关于团队合作和项目管理:**
   - 你能描述一下你在团队中的角色，以及你是如何与团队成员沟通和协作的吗？
   - 请举例说明你如何在紧迫的项目期限内管理时间和优先级？
7. **技术细节和问题解决:**
   - 请告诉我们你解决的一个技术难题或者在实习期间学到的一个重要技术概念。
   - 你在实习期间最自豪的一个技术成就是什么？
8. **对于新技术的适应和学习能力:**
   - 你是如何快速学习和适应新的技术栈或者工具的？
   - 在实习期间，你有没有自己独立学习新技术或工具来解决项目中的问题？能否给出例子？

