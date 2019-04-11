## NeuralNetworks:Representation
#### 非线性假设—神经网络的提出动机
* 无论是线性回归，还是逻辑回归，当特征太多时，计算的负荷会非常大，因此提出了神经网络。

#### 神经元和大脑—神经网络的具体提出
* 核心概念：类似于同一个大脑能够处理视觉、听觉、触觉信息，利用神经网络构建单一的学习算法自动学会如何处理不同类型的数据。

### 神经网络模型
#### 基本定义
* 神经网络定义：神经网络是大量神经元相互连接并通过电脉冲来交流的一个网络，其中神经元通过树突接收信息，通过轴突将信息传递给其他神经元
* 神经网络的组成：输入层(Input Layer)、隐藏层(Hidden Layer)、输出层(Output Layer)，每层可能会增加一个偏置单元(bias unit)

<div align = center>
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/machine-learning/picture/Neural_Network1.png" width="400" >
</div>

> $a_i^j$ 表示第 $j$ 层的第 $i$ 个激活单元\
> $\theta^j$表示第 $j$ 层映射到第 $j+1$ 层时的权重矩阵，行数为第 $j+1$ 层的激活单元，列数为第 $j$ 层的激活单元数+1

<div align = center>
<img src="https://github.com/hwt-freedom/AI/blob/master/machine-learning/picture/Neural_Network2.png" width="400" >
</div>

* 相关术语
> 神经元 -> 激活单元(activation unit)\
> 神经网络中的参数 -> 权重(weight)
