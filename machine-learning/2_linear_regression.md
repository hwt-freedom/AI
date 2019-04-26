# 线性回归
### 相关概念
* 线性回归属于 **监督学习** 的相关内容。
### 线性回归算法模型表示-单变量
#### 变量定义
* $m$ 代表训练集中实例的数量（样本个数）
* $x$ 代表特征\输入变量
* $y$ 代表目标变量\输出变量
* $(x,y)$ 代表训练集中的实例
* $(x^i,y^i)$ 代表第 $i$ 个观察实例
* $h$ 代表学习算法的解决方案\函数，通常被称为假设函数(hypothesis)
#### 模型定义
$$h_\theta(x)=\theta_0+\theta_1x$$
* $h_\theta(x)$ 为假设函数
* $\theta_0,\theta_1$ 为模型参数
* 在算法训练过程中，期望得到最优的 $\theta_0,\theta_1$ 的值使得模型最为匹配训练集的数据
#### 代价函数
* 确定模型后，需要做的是选择合适的参数 $\theta_0,\theta_1$ ，使得模型所预测的值与训练集中的实际值之间的误差最小
* 损失函数(Loss Function)：
> 将预测值与实际值的平方差或者它们平方差的一半定义为损失函数，用来衡量预测输出值与实际值有多接近。

$$L(h(x),y)=\frac{1}{2}(h(x)-y)^2$$
* 代价函数(Cost Function)
> 将所有样本的损失值求和再取平均，可以得到一个代价函数

$$J(\theta_1,\theta_0) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^i)-y^i)^2$$
* 优化目标是选择合适的模型参数 $\theta_0,\theta_1$ 使得代价函数取得最小值
* 注意：在不同的模型中，**损失函数** 和 **代价函数** 可能是不同的，如分类问题的 **损失函数** 就与当前研究的线性回归算法的 **损失函数** 和 **代价函数** 不同。
<br>
<br>
### 算法训练
* 通过手工画图计算，能够大致地观察到代价函数在何处取到最小值，但要使计算机自动地求解出代价函数 $J(\theta_0,\theta_1)$ 的最小值，需要采用有效的算法求解
* 在计算机中，**梯度下降算法** 是一个用来求函数最小值的常用算法
#### 梯度下降算法（Gradient Descent）
##### 思想：
* 开始时随机选择一个参数的组合($\theta_0,\theta_1,...,\theta_n$)，计算代价函数，然后寻找一个能让代价函数值下降的最多的参数组合，持续做直到得到一个 **局部最小值** (local minimum)
* 选择不同的初始参数组合，可能会找到不同的局部最小值，当尝试完所有的参数组合后，即可得到 **全局最小值** (global minimum)
##### 算法过程
* 不断更新 $\theta_0,\theta_1$ 参数，使得代价函数 $J(\theta_1,\theta_0)$ 收敛到一个值
$$\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1) \quad\quad (for \; j=0 \: and \:j=1)$$
* 同步更新过程
$$\begin{align}
temp0&:=\theta_0-\alpha\frac{\partial}{\partial\theta_0}J(\theta_0,\theta_1)\\
temp1&:=\theta_1-\alpha\frac{\partial}{\partial\theta_1}J(\theta_0,\theta_1)\\
\theta_0& :=temp0\\
\theta_1&:=temp1
\end{align}\\$$

##### 相关名词解释
* $\alpha$ 学习率(learning rate)：学习率决定了沿着能让代价函数下降程度最大的方向向下迈出的步子有多大，它只会影响算法收敛速度的快慢，因此选择合适的学习率很重要。
* 梯度(Gradient)：偏导数 $\frac{\partial}{\partial \theta_j}J(\theta_0,\theta_1)$ 即为点 $(\theta_0,\theta_1)$ 处的梯度值
* 同步更新(Simultaneous update)：梯度下降算法中，更新 $\theta_0,\theta_1$ 时需要同步更新两者，同步更新是一种更加自然的实现方法，在向量化实现中更加容易。

##### 算法具体实现
* 偏导数 $\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)$ 的求法
$$\begin{align}
\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1) & = \frac{\partial}{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^i)-y^i)^2\\
& =\frac{\partial}{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^{m}(\theta_0+\theta_1x^i-y^i)^2
\end{align}\\$$
* $j=0$时：
$$\begin{align}
\frac{\partial}{\partial\theta_0}J(\theta_0,\theta_1) & = \frac{\partial}{\partial\theta_0}\frac{1}{2m}\sum_{i=1}^{m}(\theta_0+\theta_1x^i-y^i)^2\\
&=\frac{1}{2m}\sum_{i=1}^{m}2(\theta_0+\theta_1x^i-y^i)
\end{align}\\$$
* $j=1$时：
$$\begin{align}
\frac{\partial}{\partial\theta_1}J(\theta_0,\theta_1) & = \frac{\partial}{\partial\theta_1}\frac{1}{2m}\sum_{i=1}^{m}(\theta_0+\theta_1x^i-y^i)^2\\
&=\frac{1}{2m}\sum_{i=1}^{m}2(\theta_0+\theta_1x^i-y^i)x^i
\end{align}\\$$

#### 参考博客
* [机器学习(一)——单变量线性回归](https://blog.csdn.net/lijiecao0226/article/details/78090453?utm_source=blogxgwz9)

### 线性回归算法模型表示-多变量
#### 多变量特征
* $m$ 代表训练集中实例的数量（样本个数）
* $x^{i}$ 代表第 $i$ 个训练样本
* $n$ 样本的特征数
* $x^{i}_j$ 代表第 $i$ 个训练样本的第 $j$ 个特征
#### 模型定义
$$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n$$
* $h_\theta(x)$ 为假设函数
* $\theta_0,\theta_1,\theta_2,...,\theta_n$ 为模型参数
* 为使得公式简洁化，定义 $x_0=1$，同时定义两个向量
$$X =\begin{bmatrix}
x_0 \\
x_1 \\
x_2 \\
 \vdots\\
 x_n
 \end{bmatrix}\quad
 \Theta =\begin{bmatrix}
 \theta_0 \\
 \theta_1 \\
 \theta_2 \\
  \vdots\\
  \theta_n
  \end{bmatrix}
 $$
 * 公式可转化为如下矩阵形式：
$$h_\theta(x)=\theta_0x_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n=X\Theta$$
#### 代价函数
* 与单变量的线性回归类似，多变量线性回归的代价函数可以表示为：
$$J(\theta_1,\theta_0,...,\theta_n) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^i)-y^i)^2$$
#### 梯度下降算法（Gradient Descent）
* 与单变量线性回归类似，只需不断求偏导，得到使代价函数最小的一系列参数
$$\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1,...,\theta_n)$$
#### 算法实现步骤
* 第一步：随机初始化 $\theta_0,\theta_1,\theta_2,...,\theta_n$
* 第二步：计算 $h_\theta(x)$ 的值
* 第三步：计算 $J(\theta_0,\theta_1,...,\theta_n)$ 的值（记录每次迭代中 $J$ 的值）
* 第四步：判断 $J(\theta_0,\theta_1,...,\theta_n)$ 是否小于 $\epsilon$ 或迭代次数是否大于阈值，若小于 $\epsilon$ 或迭代次数超过阈值则循环结束，当前 $\theta_0,\theta_1,...,\theta_n$ 为最终所求值
* 第五步：使用梯度下降公式同步更新 $\theta_0,\theta_1,...,\theta_n$
* 第六步：跳转至第二步，继续迭代
#### 梯度下降算法实践技巧
* 特征缩放：面对多维特征问题时，保证这些特征都具有相近的尺度，有助于梯度下降算法更快地收敛。
> 通常的做法是尝试将所有特征的尺度都尽量缩放到-1和1之间\
> 最简单的做法是令 $x_i= \frac{x_i-\mu_i}{s_i}$，其中 $\mu_i$ 是平均值，$s_i$ 是标准差

* 学习率：学习率影响着梯度下降算法达到收敛的次数。
> 若学习率过小，则达到收敛所需的迭代次数会非常高\
> 若学习率过大，则每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛

* 尽量通过图像来判断梯度下降算法是否正常工作
> 若梯度下降算法正常工作，则每一步迭代后 $J(\theta)$ 的值都处于下降状态

#### 特征选择和多项式回归
* 选择不同的特征得到不同的学习算法：以房屋价格预测为例，利用 $area = frontage*depth$ 实现特征的整合
$$h_\theta(x)=\theta_0+\theta_1*frontage+\theta_2*depth$$
$$h_\theta(x)=\theta_0+\theta_1*area$$
* 线性回归在拟合数据时表达能力有限，因此常常需要曲线来适应数据，比如二次方模型或三次方模型
$$二次方模型：h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2^2$$
$$三次方模型：h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2^2+\theta_3x_3^3$$

> 通常先观察数据然后再决定准备尝试何种模型

* 为了便于处理，还能够通过一定的变换将多项式回归转化为线性回归的形式
$$线性回归模型1：h_\theta(x)=\theta_0+\theta_1(size)+\theta_2(size)^2$$
$$线性回归模型2：h_\theta(x)=\theta_0+\theta_1(size)+\theta_2\sqrt{size}$$
> 采用多项式回归模型时，运行梯度下降算法之前，需要进行特征缩放

#### 正规方程
* 对于线性回归问题，也可以采取正规方程的解决方法，也称 $\theta$ 的解析解法。
* 算法思想：通过求解所有模型参数的偏导数 $\frac{\partial}{\partial\theta_j}J(\theta_j)$，令其等于0，联立方程求解，所得结果即为 $\theta$ 最优值。
##### 具体求解
$$\theta = (X^TX)^{-1}X^Ty$$
* $(x^1,y^1),...,(x^m,y^m)$ 为 $m$ 个样本
* 每个样本 $x^i$ 具有 $n$ 个特征
$$x^i =\begin{bmatrix}
x_0^i \\
x_1^i \\
x_2^i \\
 \vdots\\
 x_n^i
 \end{bmatrix} \in R^{n+1}
$$
* $X$ 为设计矩阵(design matrix)
$$X =\begin{bmatrix}
---(x^1)^T --- \\
---(x^2)^T--- \\
---(x^3)^T--- \\
 \vdots\\
 ---(x^m)^T---
 \end{bmatrix}
 $$
* $y$ 为每个样本对应的值
$$y =\begin{bmatrix}
y^1 \\
y^2 \\
y^3 \\
 \vdots\\
 y^m
 \end{bmatrix}
$$
* 可以一次性求解最优值，无需进行特征缩放
##### 正规方程的不可逆性
* 当利用正规方程求解时，会出现矩阵不可逆的情况，可能有如下两个原因导致不可逆，此时求逆运算 $(X^TX)^{-1}$ 无法运行，通常在计算机计算时会采取求伪逆的方式解决此问题。
>1、有多余的特征（通常是线性相关的特征）\
>2、特征太多（例如样本数小于特征数），可以通过删除一些特征或使用正规化方法来解决该问题。

#### 梯度下降和正规方程解法比较
##### 梯度下降算法
* 优点：当 $n$ 很大时，仍然能够运行的特别好
* 缺点：需要选择学习率 $\alpha$，需经过多次迭代
##### 正规方程算法
* 优点：无需选择学习率 $\alpha$，也无需经过多次迭代
* 缺点：需计算 $(X^TX)^{-1}$ ，当 $n$ 很大时，计算及其缓慢
##### 评价
* 通常情况下，当特征 $n$ 的值较小时，选择正规方程方法，可以获得精确值；当特征 $n$ 的值较大时，选择梯度下降方法。
* 梯度下降算法适用于各种类型的模型，正规方程方法只适用于线性模型

### 编程作业
#### 线性回归的向量化操作
*
