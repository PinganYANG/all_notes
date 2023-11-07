### sklearn & keras 函数

#### tokenizer.word_index

使用tokenizer时，tokenizer.word_index中的顺序是按照词频率排序的。因此tokenizer.texts_to_sequencess时，截取的前num_words个词就是tokenizer.word_index前num_words个词

#### Sequential() model

顺序模型，接下来可以按照增的顺序添加网络层，这些网络层可以按照顺序依次处理



#### model.add



#### model.compile



#### 正则化

##### L1

##### L2



#### from sklearn.model_selection import train_test_split

```python
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```



#### from sklearn.preprocessing import StandardScaler

#### dropout



#### Dense() keras

神经网络的全连接层，可以想象为输入和输出之间两个node层之间的连线部分

Dense(units=1, input_dim=1)

units output数

input dim 上一层输出

**【params计算为 input_dim+units，即与输入相同的w的数量和与输出相同的b的数量】**

#### Conv2D() keras

Conv2D(filters=输出的filters数量,kernel_size=(m,n)是卷积核的尺寸,strides=(1,1)是卷积在两个dim的步长,padding='valid'不padding或'same'padding用0补全)

#### MaxPool2D() keras

MaxPool2D(pool_size=(m,m)池化尺寸,strides=步长，padding='valid'不padding或'same'padding用0补全)

#### Flatten() keras

Flatten()，将cnn中的二维数据展平为一维，目的是转为一维的数据输入到Dense全连接层

#### SimpleRNN() keras

keras.layer.SimpleRNN(units=节点数，activation=激活函数默认tanh，input_shape=输入的形状(m,n))

input_shape是输入数据的维度，(m,n)代表，RNN的接受数据有两个维度：

- 第一个维度m为时间步长，即采用几个前序时间步长
- 第二个维度n是特征的数量
- 联系在一起的意义是利用m个时间步长数据来预测n个特征值

#### LSTM() keras

keras.layer.LSTM(与SimpleRNN的关键字作用相同)

#### GRU() keras

keras.layer.GRU(与SimpleRNN的关键字作用相同)



#### model.train_on_batch()



#### model.fit()



### 机器学习

#### 线性回归（keras）

##### 简单

简单的线性回归就是$y=wx+b$

那么就可以用一层一node的神经网络实现

###### 实现

```python
## -----------------------------数据集-----------------------------
X_data = np.random.rand(200)
noise = np.random.normal(loc=0,scale=0.09,size=X_data.shape)

Y_data = 2*X_data + noise

X_train, Y_train = X_data[:160], Y_data[:160]     # 前160组数据为训练数据集
X_test, Y_test = X_data[160:], Y_data[160:]       # 后40组数据为测试数据集
## -----------------------------搭模型-----------------------------
model = Sequential()
model.add(Dense(units=1, input_dim=1)) #线性的y=wx+b
model.compile(optimizer='sgd', loss='mse')

## -----------------------------训练-----------------------------
for step in range(2000):
   train_cost = model.train_on_batch(X_train, Y_train)
   if step % 100 == 0:
       print('train_cost:', train_cost)
w, b = model.layers[0].get_weights()
print('w:', w, 'b:', b)
```

##### 多元

多元线性回归的方法其实就和简单一样，只不过在模型部分的Dense改变了，units还是1，即输出一个y，input_dim为x的维度

```python
## -----------------------------搭模型-----------------------------
model = Sequential()
model.add(Dense(units=1, input_dim=n)) #线性的y=wx+b
model.compile(optimizer='sgd', loss='mse')
```

#### 非线性回归（keras）

非线性回归的主要实现方式就是在dense的线性基础上加上非线性的激活函数，这样就可以实现非线性的回归。

###### 实现

```python
## -----------------------------数据集-----------------------------
X_train = np.linspace(-1,1,200)
noise = np.random.normal(loc=0,scale=0.09,size=X_train.shape)
Y_train = X_train **3 + + noise #非线性

## -----------------------------搭模型-----------------------------
model = Sequential()
model.add(Dense(units=10, input_dim=1))
model.add(Activation('tanh'))
model.add(Dense(units=1))
model.add(Activation('tanh'))
model.summary()
sgd = SGD(lr=0.3)
model.compile(optimizer=sgd, loss='mse')

## -----------------------------训练-----------------------------
for step in range(4000):
   train_cost = model.train_on_batch(X_train, Y_train)
   if step % 500 == 0:
       print('train_cost:', train_cost)
w, b = model.layers[0].get_weights()
print('w:', w, 'b:', b)
```

两层，输入一个x然后扩展到10个node，并加上非线性的激活函数tanh，最后连接到输出y上。

SGD方法op，loss用mse



尝试多加一层后，效果没有提升





#### Logistic 回归（keras）

##### 二分类

损失就用 **二元交叉熵** binary cross-entropy

对于逻辑回归优化器就用 **Adam** 

用sigmoid实现 0 1 分类

```python
## -----------------------------数据集-----------------------------
# 假设你有一些数据
# X是具有n个特征的输入数据，y是对应的二元分类标签（0或1）
X = np.random.random((100, n))  # 示例输入数据（100个样本，每个样本n个特征）
y = np.random.randint(2, size=(100, 1))  # 示例二元分类标签（0或1）

## -----------------------------搭模型-----------------------------
# 创建Sequential模型
model = Sequential()

# 添加一个Dense层，输入维度为n，输出维度为1，使用Sigmoid激活函数
model.add(Dense(1, input_dim=n, activation='sigmoid'))

# 编译模型，使用二元交叉熵（binary cross-entropy）作为损失函数，通常对于逻辑回归使用Adam优化器
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X, y, epochs=10, batch_size=32)


## -----------------------------训练-----------------------------
# 训练模型
model.fit(X, y, epochs=10, batch_size=32)
# 模型预测
predictions = model.predict(X)

# 模型评估
loss, accuracy = model.evaluate(X, y)
```

注：上面的代码跑出来的accuracy很差，因为数据都是random出来的

##### 多分类

多分类的区别主要是：

- 激活函数改为 **softmax**
- 损失函数改为 **多类别交叉熵（categorical cross-entropy）**
- 并且对y进行one-hot coding

```python
from keras.utils import to_categorical
# 将标签进行独热编码（one-hot encoding）
y = to_categorical(y, num_classes=3)  # 如果有3个类别，使用num_classes=3

## -----------------------------搭模型-----------------------------
# 创建Sequential模型
model = Sequential()

# 添加一个Dense层，输入维度为n，输出维度为类别数量（这里是3），使用Softmax激活函数
model.add(Dense(3, input_dim=n, activation='softmax'))

# 编译模型，使用多类别交叉熵（categorical cross-entropy）作为损失函数，通常对于多分类问题使用Adam优化器
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X, y, epochs=10, batch_size=32)
```



#### Kmeans（sklearn）

kmeans的实现就直接用sklearn.cluster 中的KMeans了

KMeans(n_clusters=需要的数量, init=初始化方法，默认用kmeans++,n_init=选择centroid的种子)

```python
## -----------------------------数据集-----------------------------
from sklearn.cluster import KMeans
from keras.datasets import mnist #用mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)
# 10 clusters
n_clusters = len(np.unique(y))

## -----------------------------搭模型-----------------------------
# Runs in parallel 4 CPUs
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
# Train K-Means.

## -----------------------------训练-----------------------------

y_pred_kmeans = kmeans.fit_predict(x)
# Evaluate the K-Means clustering accuracy.

## -----------------------------评价-----------------------------
labels = kmeans.labels_
from sklearn.metrics import silhouette_score, calinski_harabasz_score
# 计算Silhouette Score
silhouette_avg = silhouette_score(x, labels)
print(f"Silhouette Score: {silhouette_avg}")
```

这个对于minst效果不太好，因为minst是图像，kmeans对处理图形不是很在行

用Silhouette Score来评价clustering的效果



#### SVM（sklearn）

##### 原理

支持向量机（SVM）是一种监督学习算法，主要用于分类和回归任务。SVM的核心思想是找到数据点之间的最优分割超平面（决策边界），以便最大化不同类别之间的边距（即分割超平面与最近数据点之间的距离）。以下是SVM的原理和关键概念：

1. 分割超平面

分割超平面是一个决策边界，用于将不同类别的数据点分开。在二维空间中，这是一条直线；在三维空间中，是一个平面；在更高维空间中，这个概念称为超平面。

2. 边距

边距是分割超平面到最近训练样本点的距离。SVM的目标是找到能够最大化这个边距的超平面。

3. 支持向量

支持向量是距离分割超平面最近的那些数据点。这些点对确定分割超平面至关重要，因为它们直接影响边距的大小。

4. 最大化边距

SVM通过优化一个目标函数来找到最大化边距的分割超平面。这个过程涉及到数学优化和拉格朗日乘子。

5. 线性与非线性

在某些情况下，数据点无法通过线性超平面有效地分割。此时，SVM可以使用所谓的核技巧，将数据映射到更高维的空间，以便在这个新空间中找到线性分割超平面。常用的核函数包括线性核、多项式核、径向基函数（RBF）核和Sigmoid核。

6. 软间隔

在实际应用中，数据可能不是完全线性可分的，或者可能存在一些异常值。为了处理这种情况，SVM引入了软间隔的概念，允许某些数据点在一定程度上违反边距规则。这通过调整正则化参数`C`来控制。

7. 数学表达

在数学上，SVM可以通过一个凸二次规划问题来表达和求解。该问题旨在最小化分割超平面的法向量的长度的平方，同时确保每个数据点至少距离分割超平面的边距的一定值。

SVM是一种强大且灵活的机器学习算法，尤其擅长处理高维数据。由于其优秀的泛化能力，它在各种机器学习任务中都得到了广泛应用。

##### 实现

```python
## -----------------------------数据集-----------------------------
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 加载数据集（例如鸢尾花数据集）
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
## -----------------------------搭模型-----------------------------
# 创建SVM分类器
svm_model = SVC(kernel='linear')  # 你可以选择不同的核函数，例如 'linear', 'poly', 'rbf', 'sigmoid'

## -----------------------------训练-----------------------------
# 训练模型
svm_model.fit(X_train, y_train)

# 进行预测
y_pred = svm_model.predict(X_test)

## -----------------------------评价-----------------------------
# 评估模型
accuracy = svm_model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
```



#### 决策树 Decision Tress（sklearn)

##### 原理

决策树是一种用于分类和回归问题的监督学习算法，它通过一系列决策规则来建立预测模型。决策树的原理基于树形结构，其中每个节点代表一个特征，每个分支代表一个特征值的测试，而每个叶子节点代表一个目标值或类别。

以下是决策树的基本原理：

1. **节点和分支**：决策树的根节点包含所有的训练数据，然后通过对特征的测试将数据分割成多个子集。每个内部节点（非叶子节点）代表一个特征，每个分支代表一个特征值的测试，决定了数据向左或向右子节点的分配。

2. **叶子节点**：叶子节点是决策树的最终节点，它包含最终的预测结果或目标值。在分类问题中，叶子节点表示一个类别；在回归问题中，叶子节点表示一个数值。

3. **决策规则**：从根节点到叶子节点的路径构成了一系列决策规则，这些规则定义了如何对新数据进行分类或回归。

4. **目标**：决策树的目标是构建一个能够根据特征来划分数据并进行准确预测的树形结构。它通过选择特征和特征值来最大化分类或回归的准确性。

5. **特征选择**：决策树的关键是选择最佳的特征来进行分割。这通常使用一些分割准则，例如信息增益、基尼不纯度或均方误差等。根据不同的准则，算法会尝试找到最佳特征和特征值来最大化分割后的纯度或准确性。

6. **树的生长和剪枝**：决策树的构建过程可以分为两个阶段，即树的生长和剪枝。在树的生长阶段，算法尝试构建尽可能深的树，使每个叶子节点包含的样本都属于同一类别或具有相似的回归值。然后，在剪枝阶段，决策树可能会被剪枝，以防止过度拟合并提高泛化性能。

7. **泛化能力**：决策树的优势之一是具有良好的可解释性和解释性，但也容易过度拟合训练数据。因此，在实际应用中，通常需要采用一些策略来优化树的结构，以平衡训练数据的拟合和模型的泛化能力。

决策树是一种强大且广泛用于机器学习和数据挖掘的算法，它们易于理解和可解释，并且在许多应用中都表现良好。不过，决策树也有一些局限性，如对噪声敏感和可能出现过度拟合问题。在实际应用中，通常需要考虑这些问题并采取相应的措施。

##### 实现

```PYTHON
## -----------------------------数据集-----------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 加载数据集（例如鸢尾花数据集）
data = load_iris()
X = data.data
y = data.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## -----------------------------搭模型-----------------------------
# 创建决策树分类器
clf = DecisionTreeClassifier()
## -----------------------------训练-----------------------------
# 训练模型
clf.fit(X_train, y_train)

# 进行预测
y_pred = clf.predict(X_test)
## -----------------------------评价-----------------------------
# 计算模型准确度
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

#### random forest 随机森林 （Bagging）（sklearn）

##### 原理

**【Bagging】【重复抽样】【有放回（利用bootstrap）】**

随机森林是一种集成学习算法，它通过结合多个决策树的预测来提高整体预测的准确性和鲁棒性。随机森林的核心思想是"群体智慧"——多个模型的综合预测通常比单个模型的预测更准确、更稳定。以下是随机森林的主要原理和步骤：

1. **集成学习**：

   - 集成学习是一种机器学习范式，它通过组合多个模型来提升整体性能。随机森林属于集成学习的一种形式，称为**Bagging**（自举汇聚法）。

2. **创建多个决策树**：

   - 在训练过程中，随机森林从原始数据集中重复抽样（有放回），为每个决策树创建一个略有不同的训练数据集。
   - 每个决策树在构建时都是相互独立的。

3. **引入随机性**：

   - 在每个决策树的构建过程中，随机森林在分割节点时不是考虑所有的特征，而是从一个随机选择的特征子集中选取最佳分割特征。
   - 这种随机特征选择增加了森林的多样性，减少了模型的方差，使其对单个决策树的过拟合更为鲁棒。

4. **预测和投票**：

   - 对于分类问题，随机森林的预测是基于所有决策树的预测结果进行多数投票的结果。
   - 对于回归问题，则是所有决策树预测结果的平均值。

5. **减少过拟合**：

   - 由于引入了随机性并且是基于多个决策树的预测，随机森林比单个决策树更不容易过拟合。

6. **特征重要性**：

   - 随机森林可以用来评估特征的重要性。通常，如果一个特征在多个树中被用来分割，且带来了显著的纯度增益，那么这个特征就被认为是重要的。

7. **优势与局限性**：

   - 优势：随机森林算法准确度高，对异常值和噪声具有较强的抵抗力，能有效处理大量输入变量，不需要特征选择，且能够评估特征的重要性。
   - 局限性：相比单个决策树，随机森林模型的可解释性较差，计算复杂度较高，训练和预测速度较慢。

随机森林由于其强大的性能和灵活性，在各种机器学习任务中得到了广泛应用，尤其适合处理高维数据和大规模数据集。

##### 实现

##### 

```PYTHON
## -----------------------------数据集-----------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据集（例如鸢尾花数据集）
data = load_iris()
X = data.data
y = data.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## -----------------------------搭模型-----------------------------
# 创建随机森林分类器
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 可以调整树的数量和其他超参数

## -----------------------------训练-----------------------------
# 训练模型
rf_model.fit(X_train, y_train)

# 进行预测
y_pred = rf_model.predict(X_test)
## -----------------------------评价-----------------------------
# 计算模型准确度
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

##### RandomForestClassifier()

RandomForestClassifier(n_estimators=决策树的数量，criterion=检测分割质量‘gini’或者‘entropy’, max_depth=树的最大深度, min_samples_split=分割结点的最少样本（少于这个数就不分割了）, min_samples_leaf=叶结点最小样本（少于这个数就不生长了）)

#### XGBoost （xgboost package）

**【Boosting】【集成学习】【正则化】【剪枝】**

##### 原理

XGBoost（Extreme Gradient Boosting）是一种高效且灵活的梯度提升框架，主要用于分类和回归问题。它是集成学习算法的一种，特别是属于Boosting算法家族。以下是XGBoost的核心原理：

1. **基于树的模型**：

   - XGBoost使用决策树作为基学习器。它逐渐添加树，每棵新树都尝试修正之前树的预测错误。

2. **梯度提升**：

   - 梯度提升是一种迭代技术，它从一个初始简单模型开始，然后逐步增加新的模型来纠正之前模型的残差（实际值与预测值之间的差距）。
   - 在每一步，XGBoost使用梯度下降算法来确定如何通过添加新的树来最大限度地减少损失函数。

3. **损失函数优化**：

   - XGBoost涉及损失函数的优化。损失函数评估预测值与实际值之间的差异。XGBoost不仅尝试最小化损失，还通过正则化项来控制模型的复杂度，防止过拟合。

4. **正则化**：

   - XGBoost在其目标函数中加入了正则化项（L1和L2正则化），这有助于减少模型的复杂度，防止过拟合。

5. **并行处理与系统优化**：

   - XGBoost对树的构建过程进行了优化，使得算法能在多核处理器上高效运行。它并不是并行构建所有树，而是在构建单个树的过程中使用并行化技术。

6. **剪枝**：

   - 与其他提升方法不同，XGBoost采用了深度优先的策略来生长树，并在达到指定最大深度后停止。然后，它会对树进行剪枝，移除不提供正向增益的分支。

7. **处理缺失值与类别特征**：

   - XGBoost能够自动处理缺失值。当存在缺失值时，XGBoost会学习该如何处理它们。
   - 它还能处理类别特征，尽管通常建议先进行编码。

8. **内置交叉验证**：

   - XGBoost允许在每一轮boosting迭代中使用交叉验证，这有助于避免过拟合。

9. **灵活性**：

   - XGBoost允许用户定义自己的优化目标和评估标准。

10. **应用广泛**：

   - 由于其灵活性、效率和强大的性能，XGBoost在各种机器学习比赛和实际应用中都非常流行。

总之，XGBoost之所以强大，不仅在于其集成了多个决策树的提升方法，还在于它引入了正则化、系统优化和多种特性来提高模型的性能和速度。

##### 实现

##### 

```PYTHON
## -----------------------------数据集-----------------------------
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 加载数据集（例如鸢尾花数据集）
data = load_iris()
X = data.data
y = data.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## -----------------------------搭模型-----------------------------
# 创建XGBoost分类器
xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, objective='multi:softmax', num_class=3)

## -----------------------------训练-----------------------------
# 训练模型
xgb_model.fit(X_train, y_train)

# 进行预测
y_pred = xgb_model.predict(X_test)

## -----------------------------评价-----------------------------
# 计算模型准确度
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

#### 

##### XGBoost和random tree的区别

**【Boosting 顺序-修正-不使用bootstrap并使用全体样本】**

**【Bagging 独立并列-投票-使用bootstrap创建不同的部分训练子集】**

XGBoost（极端梯度提升）和随机森林都是非常流行的集成学习算法，用于解决分类和回归问题，但它们在原理和功能上存在一些关键差异：

1. **算法类型**：

   - **随机森林**：它是一种**Bagging**（装袋）算法，通过构建多棵决策树并将它们的预测结果进行平均或多数投票来工作。
   - **XGBoost**：它是一种**Boosting**（提升）算法，通过顺序地添加树，每棵新树尝试纠正之前树的错误来工作。

2. **树的建立方式**：

   - **随机森林**：在随机森林中，树是独立建立的。每棵树都是从数据集的随机子样本中构建的。
   - **XGBoost**：在XGBoost中，树是顺序建立的。每棵新树使用之前所有树的知识，并专注于纠正先前树未能正确预测的那些样本。

3. **处理过拟合**：

   - **随机森林**：它通过构建多棵树并将它们的结果进行平均来减少过拟合。
   - **XGBoost**：它通过添加正则化项（L1和L2）来控制模型的复杂度，从而帮助减少过拟合。

4. **速度和性能**：

   - **随机森林**：通常比较快，因为树是并行建立的，但在某些数据集上可能不如XGBoost表现好。
   - **XGBoost**：通常在处理复杂数据集时表现更好，但需要更多的计算资源。它被优化以在速度上有所提升。

5. **参数调整**：

   - **随机森林**：相对来说参数较少，调参相对简单。
   - **XGBoost**：有更多的参数可以调整，如树的深度、学习率、正则化参数等，提供了更高的灵活性，但也需要更多的调参工作。

6. **解释性和可视化**：

   - **随机森林**：由于构建了多棵独立的树，随机森林通常提供较好的解释性。
   - **XGBoost**：虽然也能提供特征重要性的度量，但由于是顺序建树，解释性可能不如随机森林。

7. **缺失值处理**：

   - **随机森林**：通常对缺失值不太敏感。
   - **XGBoost**：能够自动处理缺失值，具有内置的策略来处理。

8. **应用场景**：

   - **随机森林**：在不需要花费大量时间调参并希望模型具有较好解释性时很有用。
   - **XGBoost**：在数据集复杂且性能至关重要的竞赛和实际应用中更常见。

总之，选择哪种算法取决于具体问题、数据集的特性以及可用的计算资源。在某些情况下，随机森林可能更为合适，而在其他情况下，XGBoost可能会提供更好的性能。



##### Boosting和Bagging的区别

**【Boosting 顺序-修正-不使用bootstrap并使用全体样本】**

**【Bagging 独立并列-投票-使用bootstrap创建不同的部分训练子集】**



#### 深度学习

##### NN （keras）

普通的NN可以利用线性回归中的方法构建

##### DNN（keras）

DNN可以利用线性回归中的方法构建，可以多加几层

##### CNN（keras）

```python
## -----------------------------导入库-----------------------------
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

## -----------------------------数据集-----------------------------
# 加载数据集（例如鸢尾花数据集）
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 4. 数据预处理
img_x, img_y = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 #归一化 这里就是说RGB的范围是0~255，然后除以255就可以让值变为0~1
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

## -----------------------------搭模型-----------------------------
# 5. 定义模型结构
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(img_x, img_y, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 6. 编译
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


## -----------------------------训练-----------------------------
# 7. 训练
model.fit(x_train, y_train, batch_size=128, epochs=10)


## -----------------------------评价-----------------------------
# 8. 评估模型
score = model.evaluate(x_test, y_test)
print('acc', score[1])
```



##### RNN（keras）

利用Keras就可以搭建一个RNN网络



```python
## -----------------------------导入库-----------------------------
## -----------------------------数据集-----------------------------
import numpy as np
import matplotlib.pyplot as plt

# 生成正弦波数据
time_steps = 1000
x = np.linspace(0, 50, time_steps)
y = np.sin(x)

def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# 设定时间步长
time_step = 5

# 准备数据集
X, y = create_dataset(y, time_step)

# 调整X的形状以适应RNN输入 (samples, time_steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

## -----------------------------搭模型-----------------------------
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

model = Sequential()
# 利用SimpleRNN建立一个简单的RNN网络
model.add(SimpleRNN(units=64, input_shape=(time_step, 1), activation='tanh'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

## -----------------------------训练-----------------------------
# 7. 训练
model.fit(X, y, epochs=100, batch_size=32, verbose=1)


## -----------------------------评价-----------------------------
# 8. 评估模型
# 进行预测
y_pred = model.predict(X)

# 可视化结果
plt.plot(y, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Sin Wave Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
```



##### LSTM（keras）

只需要将RNN模型中的SimpleRNN换为**LSTM**

##### GRU（keras）

只需要将RNN模型中的SimpleRNN换为**GRU**

##### GAN（keras）

对于GAN而言，Generator和Discriminator应该分别训练。

- 在固定生成器 训练判别器
- 先固定判别器 训练生成器

先训练

```python
## -----------------------------导入库-----------------------------
import numpy as np
from keras.layers import Dense, Reshape, Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import mnist
##------------------------------创建生成器---------------------------
def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=z_dim, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    model.add(Reshape((28, 28, 1)))
    return model
##------------------------------创建判别器---------------------------

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
##------------------------------创建GAN---------------------------
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

## -----------------------------编译和训练GAN-----------------------------
# 图像参数
img_rows = 28
img_cols = 28
img_channels = 1
img_shape = (img_rows, img_cols, img_channels)

# GAN参数
z_dim = 100

# 构建并编译判别器
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 构建生成器
generator = build_generator(z_dim)

# 构建GAN模型
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())


## -----------------------------训练GAN模型-----------------------------
def train_gan(generator, discriminator, gan, X_train, z_dim, epochs=1, batch_size=128):
    batch_count = X_train.shape[0] // batch_size

    for e in range(1, epochs + 1):
        for _ in range(batch_count):
            # 训练判别器
            noise = np.random.normal(0, 1, (batch_size, z_dim))
            generated_images = generator.predict(noise)
            real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            noise = np.random.normal(0, 1, (batch_size, z_dim))
            labels_gan = np.ones((batch_size, 1))
            g_loss = generator.train_on_batch(noise, labels_gan)

        print(f'Epoch {e}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}')

(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 127.5 - 1.0  # 归一化至[-1, 1]

train_gan(generator, discriminator, gan, X_train, z_dim, epochs=100, batch_size=128)

```



RCNN



##### Transformer (keras)

<img src="https://pic1.zhimg.com/70/v2-3b84009bb6806399e4d1b86c88bf2e23_1440w.image?source=172ae18b&biz_tag=Post" alt="TF 2.0 Keras 实现 Transformer" style="zoom:50%;" />

###### Embedding

```python
class Embedding(Layer):

    def __init__(self, vocab_size, model_dim, **kwargs):
        self._vocab_size = vocab_size # 词汇表大小，可以说是字典大小
        self._model_dim = model_dim # Embedding向量维度
        super(Embedding, self).__init__(**kwargs)

## ------------------ 构建Embedding矩阵
    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            name="embeddings")
        super(Embedding, self).build(input_shape)

## -----------------处理数据 
# K.gather 可以从嵌入矩阵中查找对应索引的嵌入向量，比如找到词 apple的嵌入向量
# 对Embedding向量进行缩放，保持其范数与模型维度成比例，便于训练
##
    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        return embeddings

## -----------------计算输出的维度
# 此方法计算嵌入层输出的形状。输入形状加上模型维度即得到输出形状。
# 这表明输出张量的最后一个维度将是嵌入向量的维度
##
    def compute_output_shape(self, input_shape):

        return input_shape + (self._model_dim,)
```

###### Positional Encoding

```python
class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)
        
## -----------------处理数据 
# 计算PositionEncoding 位置编码，方法就是论文中那个方法sin cos
##

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i-i%2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2]) # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2]) # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings
## -----------------计算输出的维度
# 输出形状与输入形状相同，因为位置编码仅添加到输入序列中，而不改变其形状
##
    def compute_output_shape(self, input_shape):
        return input_shape
```

###### Add（Embedding加上PositionEncoding）

Embedding和Position信息只需要简单相加就好

【为什么可以简单相加？？？？？？】

```python
class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)
        
## -----------------处理数据 
# 简单相加
##

    def call(self, inputs):
        input_a, input_b = inputs
        return input_a + input_b
    
## -----------------计算输出的维度
# 输出形状与输入形状相同，取第一个输入的shape
##

    def compute_output_shape(self, input_shape):
        return input_shape[0]
```

###### Scaled Dot-Product Attention 实现

multi attention的重要组成部分

```python
class ScaledDotProductAttention(Layer):
## -----------------初始化
# 初始化一些参数
##
    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        # 确定是否需要使用mask bool型 这里的mask和future不一样，这个mask 用来处理用于padding等无意义占位符
        self._masking = masking
        # 确定是否对未来信息使用mask bool型
        self._future = future
        # dropout rate
        self._dropout_rate = dropout_rate
        # 用来进行masks掩码的大负数
        self._masking_num = -2**32+1
        super(ScaledDotProductAttention, self).__init__(**kwargs)
## -----------------mask padding
# 用来处理无意义的占位符，将占位符位置赋予一个极小数
##
    def mask(self, inputs, masks):
        # 转化为浮点型，masks一般是0或1
        masks = K.cast(masks, 'float32')
        # K.tile用来将masks沿着特定轴重复，使其与inputs张量匹配
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        # 扩展masks掩码的维度
        masks = K.expand_dims(masks, 1)
        # 应用掩码，因为masks为0或1，乘以一个负数后就会作用与inputs上
        # 使这一位置在softmax中起到的作用会很小
        outputs = inputs + masks * self._masking_num
        return outputs
    
## -----------------mask future
# 用对未来信息进行mask，防止未来信息的泄露
##

    def future_mask(self, inputs):
        # 创建一个和inputs尺寸相同的单位矩阵
        diag_vals = tf.ones_like(inputs[0, :, :])
        # 将其转化为下三角矩阵，可以掩盖未来信息 .to_dense() 变为密集张量可以用于数学计算
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        # 还是用tile扩展，将下三角矩阵重复确保每一个样本input都有相同的未来掩码
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        # 创建一个与未来掩码 future_masks 相同的张量，其值与大负数相同
        paddings = tf.ones_like(future_masks) * self._masking_num
        # 应用
        # 利用tf.where实现
        # future mask为0时，则填充paddings（大负数）值
        # future mask为1时，则填充回inputs值
        # 这样就可以mask掉未来信息
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

## ----------------在一个函数中统一操作
# 
##

    
    def call(self, inputs):
        # 判断是否要进行padding mask
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs
		# 改变格式
        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')
		
        # 实现公式里的计算
        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1])) # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
        if self._masking:
            scaled_matmul = self.mask(scaled_matmul, masks) # Mask(opt.)

        if self._future:
            scaled_matmul = self.future_mask(scaled_matmul)
		# 实施softmax
        softmax_out = K.softmax(scaled_matmul) # SoftMax
        # Dropout
        out = K.dropout(softmax_out, self._dropout_rate)
        
        # 按公式点乘上values
        outputs = K.batch_dot(out, values)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
```

###### Multi-Head Attention

```python
class MultiHeadAttention(Layer):
## -----------------初始化
# 初始化一些参数
##
    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=True, future=False, trainable=True, **kwargs):
        # 多头数量
        self._n_heads = n_heads
        # 每个头的维度
        self._head_dim = head_dim
        # dropout rate
        self._dropout_rate = dropout_rate
        # 是否进行padding mask
        self._masking = masking
		# 是否进行future mask
        self._future = future
        # 权重是否可训练
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

## ----------------初始化权重
# 为queries keys values分别初始化权重
##

        
    def build(self, input_shape):
        # glorot_uniform 就是Xavier初始化
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

## ----------------在一个函数中统一操作
# 
##

    def call(self, inputs):
        # 判断是否需要padding mask
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs
        
        # 分别得到query keys 和 values的值
        queries_linear = K.dot(queries, self._weights_queries) 
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)
		
        # 分割为multi head的模式，因为有多个head注意力头
        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)
        
        # 判断是否需要padding mask
        if self._masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]
        
        # 计算attention
        attention = ScaledDotProductAttention(
            masking=self._masking, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)
		
        # 得到最终的output
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)
        
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
```

###### Position-wise Feed Forward 实现

1. **编码器中：** 在 Transformer 的编码器部分，Position-wise Feed Forward Network 被用于处理每个输入序列中每个位置的隐藏表示。编码器的每个位置都有自己的隐藏表示，这个表示包含了有关该位置的信息。Position-wise Feed Forward Network 以这个隐藏表示为输入，通过一系列的前馈神经网络层来生成新的表示，以捕获更高级的特征。
2. **解码器中：** 在 Transformer 的解码器部分，同样也使用 Position-wise Feed Forward Network 来处理每个输出序列中的每个位置。解码器的每个位置也有自己的隐藏表示，其中包含有关输出序列的信息。Position-wise Feed Forward Network 在解码器中的作用与编码器中类似，用于对每个位置的隐藏表示进行进一步的处理，以生成输出序列。

```python
class PositionWiseFeedForward(Layer):
    
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)
## ----------------初始化权重
# 分别初始化w和b
##
    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")
        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")
        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        super(PositionWiseFeedForward, self).build(input_shape)

##----------------实现前向传播操作
        
    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        return outputs

    def compute_output_shape(self, input_shape):
        return self._model_dim
```



###### Layer Normalization 实现

```python
class LayerNormalization(Layer):
## -----------------初始化
# 初始化一些参数
##
    def __init__(self, epsilon=1e-8, **kwargs):
        # 防止除以零的情况 归一化计算中的稳定项
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)
## -----------------构造
# 
##
    def build(self, input_shape):
        # 偏移参数 初始化为0
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        # 比例参数 初始化为1
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)
## -----------------实现标准化操作
# 
##
    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
```



###### Transformer 实现 包含decoder和encoder

- 包含decoder
- 包含encoder

```python
class Transformer(Layer):

    def __init__(self, vocab_size, model_dim, 
            n_heads=8, encoder_stack=6, decoder_stack=6, feed_forward_size=2048, dropout_rate=0.1, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        self._n_heads = n_heads
        # encoder数量
        self._encoder_stack = encoder_stack
        # decoder数量
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
        super(Transformer, self).__init__(**kwargs)
## -----------------构造
# 初始化Embedding
##
        
    def build(self, input_shape):

        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            trainable=True,
            name="embeddings")
        super(Transformer, self).build(input_shape)

## -----------------构造 encoder
# 
##
    def encoder(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
		# 初始化mask
        masks = K.equal(inputs, 0)
        # 建立一个 Embeddings 类似上面Embedding类 
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        # 设置 Position Encodings
        position_encodings = PositionEncoding(self._model_dim)(embeddings)
        # 设置 总encoding
        encodings = embeddings + position_encodings
        # 设置 Dropout
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._encoder_stack):
            # Multi-head-Attention
            attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            attention_input = [encodings, encodings, encodings, masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += encodings
            attention_out = LayerNormalization()(attention_out)
            # Feed-Forward
            ff = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = LayerNormalization()(ff_out)

        return encodings, masks

## -----------------构造 decoder
# 
##
    def decoder(self, inputs):
        decoder_inputs, encoder_encodings, encoder_masks = inputs
        if K.dtype(decoder_inputs) != 'int32':
            decoder_inputs = K.cast(decoder_inputs, 'int32')

        decoder_masks = K.equal(decoder_inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, decoder_inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        # Position Encodings
        position_encodings = PositionEncoding(self._model_dim)(embeddings)
        # Embedings + Postion-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)
        
        for i in range(self._decoder_stack):
            # Masked-Multi-head-Attention
            masked_attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
            masked_attention_input = [encodings, encodings, encodings, decoder_masks]
            masked_attention_out = masked_attention(masked_attention_input)
            # Add & Norm
            masked_attention_out += encodings
            masked_attention_out = LayerNormalization()(masked_attention_out)

            # Multi-head-Attention
            attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            attention_input = [masked_attention_out, encoder_encodings, encoder_encodings, encoder_masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += masked_attention_out
            attention_out = LayerNormalization()(attention_out)

            # Feed-Forward
            ff = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = LayerNormalization()(ff_out)

        # Pre-Softmax 与 Embeddings 共享参数
        linear_projection = K.dot(encodings, K.transpose(self.embeddings))
        outputs = K.softmax(linear_projection)
        return outputs

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_encodings, encoder_masks = self.encoder(encoder_inputs)
        encoder_outputs = self.decoder([decoder_inputs, encoder_encodings, encoder_masks])
        return encoder_outputs

    def compute_output_shape(self, input_shape):
        return  (input_shape[0][0], input_shape[0][1], self._vocab_size)
```



#### 画图的

plt.scatter 散点图

#### 



#### 各种方法都用在什么情况
