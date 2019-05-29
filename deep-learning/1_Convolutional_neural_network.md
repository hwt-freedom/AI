### 卷积神经网络（Convolutional Neural Network）
#### 卷积神经网络提出的目的--以计算机视觉为例
* 假如要处理64\*64\*3的图片，则特征向量 $x$ 的维度为12288
* 假如要处理1000\*1000\*3的图片，则特征向量 $x$ 的维度为300million
* 若第一个隐藏层有1000个隐藏单元，则输入层与第一个隐藏层构成的权值矩阵的维度为1000*300million，这意味着权值矩阵将有30亿个参数，此时难以获得足够的数据来防止神经网络发生过拟合的问题
> 通过卷积计算可以有效避免当前的问题

#### 边缘检测
* 垂直边缘检测中的卷积运算
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/CNN/Convolutional_neural_network1.png" width = "500">
</div>

* 垂直边缘检测的亮暗示意结果
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/CNN/Convolutional_neural_network2.png" width = "400">
</div>

* 垂直边缘检测和水平边缘检测的滤波器
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/CNN/Convolutional_neural_network3.png" width = "400">
</div>

* 卷积神经网络将滤波器作为学习的参数，训练神经网络的目的就是学习这些参数

#### padding操作
##### padding的提出--解决原始卷积操作的缺陷
* 缺陷1：每次进行卷积操作时，图像都会缩小，假设输入图像为 $n*n$ ，用 $f*f$ 的过滤器做卷积，则输出的维度是 $(n-f+1)*(n-f+1)$
* 缺陷2：图像的边缘像素在卷积操作时较少使用，会丢掉许多图像边缘位置的信息
##### padding的类型
* valid padding：表示不填充，输入图像为 $n*n$ ，输出缩小。
> 用 $f*f$ 的过滤器做卷积，则输出的维度是 $(n-f+1)*(n-f+1)$
* same padding：表示填充后，输出的大小和输入大小相同。
> 输入图像为 $n*n$，填充 $p$ 个像素点后，图像变为 $n+2p$\
> 输出图像为 $(n+2p-f+1)*(n+2p-f+1)$\
> 令 $n+2p-f+1=n$ ，有 $p=\frac{f-1}{2}$

#### 卷积步长（strided convolutions）
* 控制卷积时滤波器移动的步长，此时卷积图像的大小计算公式为
> 输入图像为 $n*n$，过滤器为 $f*f$，步长为 $s$，padding为p\
> 输出图像为 $\lfloor(\frac{(n+2p-f)}{2}+1)*(\frac{(n+2p-f)}{2}+1)\rfloor$

#### 三维卷积
##### RGB图像卷积
* RGB图像卷积操作示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/CNN/Convolutional_neural_network4.png" width = "400">
</div>

> 图像的通道数必须和滤波器的通道数相匹配\
> 如果只考虑某个颜色通道，可以将滤波器的其他两个通道置0
##### 多滤波器--卷积神经网络重要思想
* 假如希望同时检测垂直边缘和水平边缘或者其他任意边缘时，需要采用多个过滤器，输入输出之间遵循如下关系
> 输入图像 $n*n*n_c$，滤波器 $f*f*n_{c_0}$，且滤波器的个数为$n_{c_1}$\
> 步长为 $s$，padding为p\
> 输出图像为 $\lfloor(\frac{(n+2p-f)}{2}+1)*(\frac{(n+2p-f)}{2}+1)\rfloor*n_{c_1}$
* 输出的通道数等于检测的特征数，文献中也常将通道数称为深度（注意避免和神经网络的深度混淆）

#### 卷积神经网络的构成
* 卷积层(convolution,conv)
* 池化层(pooling,pool)
* 全连接层(Fully connected,FC)
> 设计重点：滤波器的尺寸、步长，padding的数目，滤波器的个数

#### 卷积层的设计
* 以上一小节的三维卷积为例，图像通过两个过滤器得到了两个 $4*4$ 的矩阵，在两个矩阵上分别加入偏差 $b_1$ 和 $b_2$ ，然后对加入偏差的矩阵做非线性的 $Relu$ 变换，得到两个新的 $4*4$ 矩阵，具体用公式表示如下：
$$z^{[1]}=w^{[1]}a^{[0]}+b^{[1]}$$
$$a^{[1]}=g(z^{[1]})$$
> 其中输入图像为 $a^{[0]}$，滤波器用 $w^{[1]}$ 表示\
> $z^{[1]}$ 表示对图像进行线性变化并加入偏差得到的矩阵\
> $a^{[1]}$ 是应用Relu函数激活后的结果

* 卷积作用的核心思想描述：假如滤波器的维度是 $3*3*3$ ，再加上一个偏差系数，共有10个滤波器，则总的参数是280个，无论输入的图片尺寸有多大，参数始终是280个
* 以卷积神经网络的第 $l$ 层为例，描述该层的各种标记
> $f^l$ ：第l层滤波器的尺寸\
> $p^l$ ：第l层padding的数量\
> $s^l$ ：第l层步长的大小\
> $n^l_C$ ：滤波器的个数\
> 输入：$n^{l-1}_H*n^{l-1}_W*n^{l-1}_C$ ：l-1层输入图像的高、宽以及通道数\
> 输出：$n^{l}_H*n^{l}_W*n^{l}_C$ ：第l层输出图像的高、宽以及通道数\
> $n^l_H=\lfloor\frac{n^{l-1}_H+2*p^l-f^l}{s^l}+1\rfloor$\
> $n^l_W=\lfloor\frac{n^{l-1}_W+2*p^l-f^l}{s^l}+1\rfloor$

#### 池化层设计
* 池化的作用：缩减模型大小，提高提取特征的鲁棒性
* 池化的类型：最大池化、平均池化
* 最大池化示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/CNN/Convolutional_neural_network5.png" width = "500">
</div>

> 最大池化比平均池化更为常用，池化的参数有矩阵的尺寸 $f$ 和步长 $s$\
> 池化层输出图像的计算公式与卷积层相同，$\lfloor(\frac{(n+2p-f)}{2}+1)*(\frac{(n+2p-f)}{2}+1)\rfloor*n_{c_1}$\
> 池化层没有要学习的参数，只是一个静态的属性

#### 卷积神经网络示例
* 典型CNN示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/CNN/Convolutional_neural_network6.png" width = "500">
</div>

* 各层参数统计示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/CNN/Convolutional_neural_network7.png" width = "500">
</div>

* 卷积层的参数相对较少
* Pooling层不包含参数
* 绝大部分参数集中在全连接层
* 激活单元随着网络的加深而逐渐减小
> 如何更加高效地构建网络，需要大量阅读已有的案例

#### 为何使用卷积--以图像边缘检测为例
* 与全连接层相比，卷积层的两个主要优势在于参数共享和稀疏连接
* 示例：假如相邻两层之间的图片维度为 $32*32*3$ 和 $28*28*6$，若采用全连接的方式构建神经网络，则其中共有将近1400万个参数；若采用卷积的方式构建神经网络，则需要的过滤器尺寸都为 $5*5$ ，再加上偏差参数，6个过滤器总共只有 $156$ 个参数。
* 参数共享：对于边缘检测，整张图片可以共享特征检测器
* 稀疏连接：在每一层的每个输出值都取决于滤波器范围内的小部分输入值，其他像素值不会对输出产生任何影响，即可以视为稀疏连接
> 卷积神经网络通过以上两种方式减少参数，使得能够通过更小的训练集进行训练，预防过度拟合
