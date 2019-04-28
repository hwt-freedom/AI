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
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/Convolutional_neural_network2.png" width = "400">
</div>

* 垂直边缘检测和水平边缘检测的滤波器
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/Convolutional_neural_network3.png" width = "400">
</div>

* 卷积神经网络将滤波器作为学习的参数，训练神经网络的目的就是学习这些参数

#### padding操作
##### padding的提出--解决原始卷积操作的缺陷
* 缺陷1：每次进行卷积操作时，图像都会缩小，假设输入图像为 $n*n$ ，用 $f*f$ 的过滤器做卷积，则输出的维度是 $(n-f+1)*(n-f+1)$
* 缺陷2：图像的边缘像素在卷积操作时较少使用，会丢掉许多图像边缘位置的信息
##### padding的类型
* valid卷积：表示不填充，输入图像为 $n*n$ ，输出缩小。
> 用 $f*f$ 的过滤器做卷积，则输出的维度是 $(n-f+1)*(n-f+1)$
* same卷积：表示填充后，输出的大小和输入大小相同。
> 输入图像为 $n*n$，填充 $p$ 个像素点后，图像变为 $n+2p$\
> 输出图像为 $(n+2p-f+1)*(n+2p-f+1)$\
> 令 $n+2p-f+1=n$ ，有 $p=\frac{f-1}{2}$

#### 卷积步长（strided convolutions）
* 控制卷积时滤波器移动的步长，此时卷积图像的大小计算公式为
> 输入图像为 $n*n$，过滤器为 $f*f$，步长为 $s$，padding为p\
> 输出图像为 $\lfloor\frac{(n+2p-f+1)}{2}*\frac{(n+2p-f+1)}{2}\rfloor$

#### 三维卷积
##### RGB图像卷积
* RGB图像卷积操作示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/Convolutional_neural_network4.png" width = "400">
</div>

> 图像的通道数必须和滤波器的通道数相匹配\
> 如果只考虑某个颜色通道，可以将滤波器的其他两个通道置0
##### 多滤波器--卷积神经网络重要思想
* 假如希望同时检测垂直边缘和水平边缘或者其他任意边缘时，需要采用多个过滤器，输入输出之间遵循如下关系
> 输入图像 $n*n*n_c$，滤波器 $f*f*n_{c_0}$，且滤波器的个数为$n_{c_1}$\
> 步长为 $s$，padding为p\
> 输出图像为 $\lfloor\frac{(n+2p-f+1)}{2}*\frac{(n+2p-f+1)}{2}\rfloor*$n_{c_1}$$
* 输出的通道数等于检测的特征数，文献中也常将通道数称为深度（注意避免和神经网络的深度混淆）

#### 单层卷积网络
* 以上一小节的三维卷积为例，图像通过两个过滤器得到了两个 $4*4$ 的矩阵，在两个矩阵上分别加入偏差 $b_1$ 和 $b_2$ ，然后对加入偏差的矩阵做非线性的 $Relu$ 变换，得到两个新的 $4*4$ 矩阵，具体用公式表示如下：
$$z^{[1]}=w^{[1]}a^{[0]}+b^{[1]}$$
$$a^{[1]}=g(z^{[1]})$$
> 其中输入图像为 $a^{[0]}$，滤波器用 $w^{[1]}$ 表示\
> $z^{[1]}$ 表示对图像进行线性变化并加入偏差得到的矩阵\
> $a^{[1]}$ 是应用Relu函数激活后的结果
