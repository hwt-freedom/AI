### 卷积神经网络（Convolutional Neural Network）
#### 卷积神经网络提出的目的--以计算机视觉为例
* 假如要处理64\*64\*3的图片，则特征向量 $x$ 的维度为12288
* 假如要处理1000\*1000\*3的图片，则特征向量 $x$ 的维度为300million
* 若第一个隐藏层有1000个隐藏单元，则输入层与第一个隐藏层构成的权值矩阵的维度为1000*300million，这意味着权值矩阵将有30亿个参数，此时难以获得足够的数据来防止神经网络发生过拟合的问题
> 通过卷积计算可以有效避免当前的问题

#### 边缘检测
* 垂直边缘检测中的卷积运算
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/Convolutional_neural_network1.png" width = "500">
</div>

* 垂直边缘检测的亮暗示意结果
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/Convolutional_neural_network2.png" width = "500">
</div>

* 垂直边缘检测和水平边缘检测的滤波器
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/Convolutional_neural_network3.png" width = "500">
</div>

* 卷积神经网络将滤波器作为学习的参数，训练神经网络的目的就是学习这些参数
####
