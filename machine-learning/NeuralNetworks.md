## NeuralNetworks:Representation
#### 非线性假设—神经网络的提出动机
* 无论是线性回归，还是逻辑回归，当特征太多时，计算的负荷会非常大，因此提出了神经网络。

#### 神经元和大脑—神经网络的具体提出
* 核心概念：类似于同一个大脑能够处理视觉、听觉、触觉信息，利用神经网络构建单一的学习算法自动学会如何处理不同类型的数据。

### 神经网络模型
#### 基本定义
* 神经网络定义：神经网络是大量神经元相互连接并通过电脉冲来交流的一个网络，其中神经元通过树突接收信息，通过轴突将信息传递给其他神经元
* 神经网络的组成：输入层(Input Layer)、隐藏层(Hidden Layer)、输出层(Output Layer)，每层可能会增加一个偏置单元(bias unit)

<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/machine-learning/picture/Neural_Network1.png" width = "400">
</div>

> $a_i^j$表示第 $j$ 层的第 $i$ 个激活单元\
> $\theta^j$表示第 $j$ 层映射到第 $j+1$ 层时的权重矩阵，行数为第 $j+1$ 层的激活单元，列数为第 $j$ 层的激活单元数+1

* 以第二层为例，激活单元和输出之间的关系可以表示为
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/machine-learning/picture/Neural_Network2.png" width="400" >
</div>

> 通过将得到的 $a_1^2,a_2^2,a_3^2$ 作为新特征，可以得到更为复杂的假设函数\
> 这种从前往后逐层计算的方法被称为前向传播算法(Forward Propagation)

* 相关术语
> 神经元 -> 激活单元(activation unit)\
> 神经网络中的参数 -> 权重(weight)

#### 前向传播算法(Forward Propagation)的向量化实现
* 以介绍过的三层网络为例，$z_i^j$ 表示计算第 $j$ 层第 $i$ 个神经元sigmoid函数的输入值，例如 $z_1^2$ 可以用如下式子表示
 $$z_1^2=\theta_{10}^1x_0+\theta_{11}^1x_1+\theta_{12}^1x_2+\theta_{13}^1x_3$$
 * $X$ 表示特征矩阵，$z^j$ 表示第 $j$ 层所有神经元sigmoid函数输入值构成的矩阵
$$
X =\begin{bmatrix}
x_0 \\
x_1 \\
x_2 \\
x_3
 \end{bmatrix}\quad
 z^2 =\begin{bmatrix}
 z_1^2 \\
 z_2^2 \\
 z_3^2 \\
  \end{bmatrix}
 $$
* $\Theta^j$ 表示第$j$层到第$j+1$层映射的权重矩阵
> 神经网络的第 $j$ 层有 $s_j$ 个单元，第 $j+1$ 层有 $s_{j+1}$ 个单元\
>
$$z^2=\Theta^1X^T$$
