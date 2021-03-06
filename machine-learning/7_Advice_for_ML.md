### 应用机器学习的建议
#### 算法改进
##### 模型诊断
* 当训练的算法面对新的测试集效果不佳时，此时需要具体问题具体分析，可以考虑运用如下几种改进方法
>-尝试获得更多的训练样本\
>-尝试减少特征的数量\
>-尝试获取附加的特征\
>-尝试增加多项式的特征\
>-尝试增加 $\lambda$\
>-尝试减小 $\lambda$

##### 评估一个假设
* 将数据集分成训练集和测试集，利用训练集训练出的参数用测试集数据测试性能
>通常情况下，训练集包括70%的数据，测试集是剩下的30%数据
* 评估假设的步骤
>1、学习 $\Theta$ 并使用训练集最小化 $J_{test}(\Theta)$\
>2、计算测试集错误 $J_{test}(\Theta)$
* 线性回归测试集获得代价函数
$$J_{test}(\Theta) = \frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}(h_\Theta(x_{test}^i)-y_{test}^i)^2$$
* 逻辑回归测试集获得代价函数
$$J_{test}(\theta) = -\frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}[y_{test}^ilog(h_\theta(x_{test}^i))+(1-y_{test}^i)log(1-h_\theta(x_{test}^i))]$$
逻辑回归还能通过计算误分类的比例来对假设进行评估

##### 模型选择和交叉验证集
* 模型选择的理由：假如学习算法只是很好地适合训练集，并不意味着这是一个很好的假设，其可能会过于合适导致其对测试集的预测会很差；假如学习算法很好地适合训练集，同时也适合测试集，只意味着得到的模型只是对泛化误差的乐观估计，依然无法保证在面对新问题时拥有较好的预测性能。
* 解决方法：使用60%的数据作为训练集，20%的数据作为交叉验证集，使用20%的数据作为测试集。
* 模型选择的步骤：
> 1、使用训练集训练出10个模型\
> 2、用10个模型分别对交叉验证集计算得出交叉验证误差（代价函数的值）\
> 3、选取代价函数值最小的模型\
> 4、从步骤3中选出的模型对测试集计算得出泛化误差（代价函数的值）

* 训练误差计算公式
$$J_{train}(\theta)=\frac{1}{2m_{train}}\sum_{i=1}^{m_{train}}(h_\theta(x^i)-y^i)^2$$
* 交叉验证误差计算公式
$$J_{cv}(\theta)=\frac{1}{2m_{cv}}\sum_{i=1}^{m_{cv}}(h_\theta(x^i)-y^i)^2$$
* 测试误差计算公式
$$J_{test}(\theta)=\frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}(h_\theta(x^i)-y^i)^2$$

#### 偏置与方差
##### 诊断偏差与方差
* 高偏差、恰好、高方差示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/machine-learning/picture/7_Advice_for_ML/1.png" width = "400">
</div>

> 高偏差(high bias)：欠拟合\
> 高方差(high variance)：过拟合

* 训练集和交叉验证集的代价函数误差与多项式的次数关系
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/machine-learning/picture/7_Advice_for_ML/2.png" width = "400">
</div>

> 随着多项式阶数d的增加，训练误差将趋于减小\
> 随着多项式阶数d的增加，交叉验证误差将先减小，后增加，形成凸曲线\
> 转折点是模型开始 **拟合** 训练数据集的时候

* 当交叉验证集误差较大时，诊断此时是高偏差还是高方差的方法：
> 高偏差(underfitting)： $J_{train}(\theta)$ 和 $J_{cv}(\theta)$ 都很高\
> 高方差(overfitting)：$J_{train}(\theta)$ 较低，$J_{cv}(\theta)$ 较高

##### 正则化方差与偏差
* 同时考虑方差与偏差，选择合适的正则化参数 $\lambda$ 值
* 选择 $\lambda$ 的方法：
> 1，使用训练集训练出12(或更多)个不同程度正则化的模型\
> 2，用12个模型分别对交叉验证集计算得出交叉验证误差\
> 3，选择得出交叉验证误差最小的模型\
> 4，运用步骤3中选出模型对测试集计算得出泛化误差，判断其是否具有良好的问题概括性
* 训练集误差和交叉验证集误差随 $lambda$ 的变化示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/machine-learning/picture/7_Advice_for_ML/3.png" width = "400">
</div>

##### 学习曲线(Learning Curves)
* 利用学习曲线判断算法是否处于偏差或方差的问题
* 高偏差/欠拟合时的学习曲线
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/machine-learning/picture/7_Advice_for_ML/4.jpg" width = "400">
</div>

* 高方差/过拟合时的学习曲线
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/machine-learning/picture/7_Advice_for_ML/5.jpg" width = "400">
</div>

##### 决定下一步做什么
* 尝试获得更多训练实例：高方差
* 尝试减少特征的数量：高方差
* 尝试获得更多的特征：高偏差
* 尝试增加多项式特征：高偏差
* 尝试减少正则化参数 $lamda$ ：高偏差
* 尝试增加正则化参数 $lamda$ ：高方差

##### 诊断神经网络
* 较小的神经网络，类似于参数较少的情况，容易导致高偏差和欠拟合，但计算复杂度较小
* 较大的神经网络，类似于参数较多的情况，容易导致高方差和过拟合，虽然计算复杂度较大，但还是可以通过正则化的方法调整使得更加适应数据
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/machine-learning/picture/7_Advice_for_ML/6.jpg" width = "400">
</div>

> 通常选择较大的神经网络并采用正则化处理会比采用较小的神经网络效果更好
* 选择神经网络的建议：从一层开始逐渐增加层数，选择交叉验证集误差最小的神经网络模型。
