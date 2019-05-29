### 循环神经网络（Recurrent Neural Network）
#### 序列模型
* 特点：输入或输出都是序列，呈时序关系
* 典型应用：语音识别、音乐生成、情感分析、DNA序列分析、机器翻译、视频行为识别、人名实体识别等

#### 数学符号（Notation）————以命名实体识别系统为例
假设要建立一个序列模型，其输入语句如下：
> x : Harry Potter and Herminoe Granger invented a new spell
* $x^{<t>}$：表示输入第 $t$ 个单词
* $T_x$：表示输入的单词总数，此时 $T_x=9$
* $y^{<t>}$：表示输出的第 $t$ 个单词
* $T_y$：表示输出总数，此时 $T_y=9$

对于第 $i$ 个样本，则表示如下：
* $x^{i<t>}$：表示第 $i$ 个样本输入的第 $t$ 个单词
* $T^i_x$：表示第 $i$ 个样本输入的单词总数，此时 $T_x=9$
* $y^{i<t>}$：表示第 $i$ 个样本输出的第 $t$ 个单词
* $T^i_y$：表示第 $i$ 个样本的输出总数，此时 $T_y=9$

#### 用one-hot向量表示一个单词
* 首先建立一个词汇表 vocabulary
* one-hot向量表示单词：每个单词用一个和词表长度一样的向量来表示，向量中除了这个单词所在位置为1，其余全都为0
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN1.png" width = "650">
</div>

> 例如：$x^{<1>}$ 表示Harry这个单词，由于在vocabulary中，其位置在第1075行，因此 $x^{<1>}$ 是一个第1075行是1，其余值都是0的向量
* 利用one-hot表示方法表示输入X，用序列模型在输入X和目标输出Y之间建立一个映射
> 如果遇到一个不在词表中的单词，则可以创建一个新标记，记为 $<Unk>$


#### 循环神经网络模型
##### RNN提出的原因
* 在解决序列学习问题时，标准神经网络存在的问题：
1. 输入和输出在不同的样本中可能有不同的长度，即使能先找到输入输出的最大值，再对其他样本进行填充，但这种表示方式依然不够好
2. 无法共享特征，在文本某一位置学习得到的特征无法应用于其他位置，并且特征数量巨大，使得第一层的权重矩阵参数的数目巨大

##### RNN具体模型
* 首先将第一个单词 $x^{<1>}$ 输入神经网络，并预测 $y^{<1>}$
* 再将第二个单词 $x^{<2>}$ 输入神经网络，同时将第一步计算的激活值 $a^{<1>}$ 也输入到神经网络中，共同预测 $y^{<2>}$
* 重复第二步，直到所有单词都训练完毕
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN2.jpg" width = "500">
</div>

> 在零时刻需要构造一个激活值 $a^{<0>}$，通常使用零向量作为零时刻的伪激活值输入神经网络
* RNN从左到右扫描数据，每个时间步的参数也是 **共享** 的，从输入到隐藏层的参数表示为 $W_{ax}$ ，水平方向激活值向下一层输入的参数表示为 $W_{aa}$，从隐藏层到输出层的参数表示为 $W_{ya}$

#### 前向传播(Forward Propagation)
* 输入一个零向量 $a^{<0>}$
* 先计算激活值 $a^{<1>}$ ，再计算 $y^{<1>}$ ：
$$a^{<1>}=g_1(W_{aa}a^{<0>}+W_{ax}x^{<1>}+b_a)$$
$$\hat{y}^{<1>}=g_2(W_{ya}a^{<1>}+b_y)$$
* RNN中的激活函数通常是 $tanh$ ，也可以是 $Relu$ ，选用哪个激活函数取决于输出 $y$
>更一般的情况，在 $t$ 时刻
$$a^{<t>}=g_1(W_{aa}a^{<t-1>}+W_{ax}x^{<t>}+b_a)$$
$$\hat{y}^{<t>}=g_2(W_{ya}a^{<t>}+b_y)$$
* 进一步简化符号，将矩阵 $W_{aa}$ 与矩阵 $W_{ax}$ 水平并列放置
$$[W_{aa}\vdots W_{aw}]=W_a$$
* 则此时
$$g_1(W_{aa}a^{<t-1>}+W_{ax}x^{<t>}+b_a)=g_1(W_a[a^{<t-1>},x^{<t>}]+b_a)$$
* 同理对于 $\hat{y}$ ，也用更简洁的形式表示
$$\hat{y}^{<t>}=g_2(W_{ya}a^{<t>}+b_y)=g_2(W_{y}a^{<t>}+b_y)$$
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN3.jpg" width = "500">
</div>

#### 通过时间的反向传播(Backpropagation through time)
* 反向传播过程示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN4.jpg" width = "500">
</div>

##### 损失函数
* 元素损失函数
$$L^{<t>}(\hat{y}^{<t>},y^{<t>})=-y^{<t>}log\hat{y}^{<t>}-(1-\hat{y}^{<t>})log(1-\hat{y}^{<t>})$$
> $y^{<t>}$ 对应序列中一个具体的词，若是某个人的名字，则 $y^{<t>}$ 的值为1，然后神经网络将输出这个词是名字的概率值， $\hat{y}^{<t>}$ 的值可以是0.1
* 序列损失函数
$$L(\hat{y},y)=\sum^{T_x}_{t=1}L^{<t>}(\hat{y}^{<t>},y^{<t>})$$
> 对每个单独时间步的损失函数求和
* 通过反向传播算法在相反的方向上进行计算和传递信息，最终就是将前向传播的箭头反过来并计算得到所有合适的量，再通过 **导数** 相关的参数，用梯度下降法来更新参数
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN5.jpg" width = "500">
</div>

#### 不同结构的循环神经网络
##### 多对一（情感分类）
* 情感分类问题：$x$ 输入一个序列，$y$ 可能是一个数字，RNN读入整个句子，在最后一个时间上得到输出
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN6.png" width = "250">
</div>

##### 一对多（音乐生成）
* 音乐生成问题：$x$ 输入想要的音乐类型或想要音乐的第一个音符，甚至是输入零向量，$y$ 为一段音乐
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN7.png" width = "250">
</div>

##### 多对多（机器翻译）
* 机器翻译问题：$x$ 输入句子单词的数量，比如说一个法语的句子，$y$ 输出句子单词的数量，比如说翻译成英语，两个句子的长度可能不同
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN8.png" width = "250">
</div>

##### 结构总结
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN9.png" width = "500">
</div>


#### 语言模型和序列生成（Language model and sequence generation）
##### 语言模型
* 语言模型的定义：得到某个特定句子出现的概率，估计某个句子序列中各个单词出现的可能性，进而给出整个句子出现的可能性。
##### 利用RNN生成语言序列
* 首先需要一个训练集，其包含一个很大的英文文本语料库(corpus)
* 标识化，将输入的句子映射到各个标志上
> Tokenize：将句子使用字典库标记化，将每个单词都转换成对应的one-hot向量\
> EOS：句子的结尾用EOS标记\
> UNK：未出现在字典库中的词使用UNK表示
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN10.jpg" width = "500">
</div>

* RNN模型建立过程
> 使用零向量对输出进行预测，输入RNN，即由softmax分类器预测第一个单词是某个单词的可能性，出现概率最高的词即为输出的词，记作 $\hat{y}^{<1>}$ 其中softmax输出个数与词汇表vocabulary个数相同\
> 再将 $\hat{y}^{<1>}$ 赋值给 $x^{<2>}$ ，再次输入RNN，获得在 $\hat{y}^{<1>}$ 情况下，第二个时间步的预测值 $y^{<2>}$\
> 依次类推，重复步骤直到句尾
* 利用softmax损失函数计算loss，更新网络参数，提升语言模型的准确率
$$L(\hat{y},y)=\sum^{T_x}_{t=1}L^{<t>}(\hat{y}^{<t>},y^{<t>})$$
> 将每一个softmax层输出概率相乘，即可得到整体句子出现的概率
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN11.jpg" width = "500">
</div>

#### 对新序列采样（Sampling novel sequence）
* 前提：已得到训练完毕的RNN
* 主要思想：与RNN生成语言序列类似，不同之处得到softmax层输出之前，对其概率分布进行随机采样，并且无论采样得到词是什么，都将其传递到下一个位置作为输入，softmax层预测输出之前继续进行随机采样，依此类推
* 使采样停止的两种方法：
> 当采样得到句子的结束标志符EOS时\
> 自行设置结束的时间步
* 对于UNK标志的词，如果不希望在结果中出现，可以在采样过程中当出现UNK标志时，对剩余词进行重采样，确保输出时没有UNK标志
* 采样过程示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN12.jpg" width = "500">
</div>

#### 基于字符的RNN语言模型
* 主要思想：基于字符，将所有可能出现的字符加入字典，例如a~z的字符，也可以包括空格、标点符号、数字，若需要区分大小写，还可以加上字符A~Z
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN13.jpg" width = "500">
</div>

* 优势：不用担心出现未知标识
* 缺点：得到的序列通常很长很多，不善于捕捉句子中的关系，捕捉范围也比基于单词的RNN短，计算成本较高
> 目前较为常见的依旧是基于词汇的语言模型，但随着计算机运算能力的增强，在某些特定的情况下，也会开始使用基于字符的语言模型

#### RNN的梯度消失
* 目前存在的问题：对于英文文本而言，可能会出现句子主语和系动词距离比较远的情况，即后面的词对前面的词有长期的依赖关系，但目前RNN的基本结构难以捕捉这种长期依赖关系。并且由于 **梯度消失（vanishing gradients）** 或 **梯度爆炸（exploding gradients）** 问题，两个相距很远的隐藏层很难相互影响。
##### 梯度爆炸（exploding gradients）
* 基本概念：指数级大的梯度会使得参数变得极其大，以至于网络参数完全崩溃，网络计算出现数值溢出，由于参数很大，因此容易发现问题
* 解决：梯度修剪，即最大值修剪。观察梯度向量，一旦其超过某个阈值，就对其缩放，以保证其不会过大
##### 梯度消失（vanishing gradients）
* 基本概念：两个间隔较远的隐藏层难以相互影响
* 解决：长短时记忆网络LSTM，门控循环单元GRU

#### Gated Recurrent Unit（门控循环单元 GRU）
##### RNN单元的可视化呈现
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN14.jpg" width = "500">
</div>

##### GRU单元核心思想
* 核心思想：相比于普通的RNN，GRU增加了一个记忆细胞C（memory cell），其提供了长期的记忆能力
* 根据实际需要记忆的内容决定隐藏激活值的维度，并在每一个时间步对记忆细胞进行更新

##### 简化的GRU单元
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN15.png" width = "500">
</div>

##### 完整的GRU单元
* 完整GRU单元模型
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN16.jpg" width = "350">
</div>

* GRU单元计算公式
$$\hat{c}^{<t>}=tanh(W_c[\Gamma_r*c^{<t-1>},x^{<t>}]+b_c)$$
$$\Gamma_u=\delta(W_u[c^{<t-1>},x^{<t>}]+b_u)$$
$$\Gamma_r=\delta(W_r[c^{<t-1>},x^{<t>}]+b_r)$$
$$a^{<t>}=c^{<t>}$$
> $c^{<t>}$ ：记忆细胞在t时间步的值，表示是否要记忆此处的信息，GRU中 $c^{<t>}=a^{<t>}$\
> $\hat{c}^{<t>}$ ：重写记忆细胞的候选值，是否采用这个值来更新 $c^{<t>}$ 取决于 $\Gamma_u$\
> $\Gamma_r$ ：相关门，表示$c^{<t-1>}$ 与新的候选值 $\hat{c}^{<t>}$ 之间的相关性\
> $\Gamma_u$ ：更新update门，决定何时更新记忆细胞，取值范围在0~1，0表示不更新记忆细胞即保持之前的值，1表示更新记忆细胞为候选值

* $\Gamma_u$ 的值要么特别小，要么无限接近于1，即使经过很多时间步，也依然能够很好地维持记忆细胞的值，因此梯度消失问题得到解决，神经网络能够运行在非常庞大的依赖词上

#### Long Short Term Memory(长短时记忆 LSTM) Unit
* 核心思想：比GRU更为强大和通用，$a^{<t>}$ 和 $c^{<t>}$ 不再相等，具有更新门、输出门和遗忘门
##### LSTM模型
* 模型可视化示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN17.jpg" width = "500">
</div>

* LSTM单元可视化示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN18.jpg" width = "500">
</div>

* LSTM单元计算公式
$$\hat{c}^{<t>}=tanh(W_c[a^{<t-1>},x^{<t>}]+b_c)$$
$$\Gamma_u=\delta(W_u[a^{<t-1>},x^{<t>}]+b_u)$$
$$\Gamma_f=\delta(W_f[a^{<t-1>},x^{<t>}]+b_f)$$
$$\Gamma_o=\delta(W_o[a^{<t-1>},x^{<t>}]+b_o)$$
$$c^{<t>}=\Gamma_u*\hat{c}^{<t>}+\Gamma_f*c^{<t-1>}$$
$$a^{<t>}=\Gamma_o*c^{<t>}$$

* 只要正确设置遗忘门和更新门的值，LSTM就可以非常容易将记忆细胞的值一直往后面的时间步传递，即使经过很长的时间步，也能够记忆某个值，因此克服梯度消失问题。
* 某些LSTM模型还会进一步添加“窥视孔连接”(peephole connection)，主要思想是门值不仅取决于 $a^{<t-1>}$ 和 $x^{<t>}$ ，也取决于上一个记忆细胞的值 $c^{<t-1>}$
##### GRU和LSTM比较
* GRU结构简单，有利于构建深层RNN
* LSTM功能强大，更为灵活

#### 双向循环神经网络(Bidirectional RNN)
* 提出原因：对于某处的序列，不仅仅需要获取之前的信息，还可以获取未来的信息
* BRNN结构示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN19.jpg" width = "400">
</div>

* 对于时间步 $t$ ，得到正向和反向激活值后，可以利用下述公式预测结果
$$\hat{y}=g(W_g[\overleftarrow{a}^{<t>},\overrightarrow{a}^{<t>}]+b_y) $$

> 其中各单元既可以是标准RNN单元，也可以是GRU、LSTM单元，目前对于NLP问题来说基于LSTM单元的BRNN使用更为广泛
* BRNN缺陷：只有得到完整序列的数据才能预测。例如对于语音识别而言，必须等说话的人说完，才能够进行预测，因此BRNN并不适用于语音识别，其通常应用于NLP问题

#### 深度RNN
* 对于RNN而言，由于时间的维度足够大，因此不同于标准神经网络或卷积神经网络那样拥有大量隐含层，RNN堆叠达到3层就已经足够大，若需要堆叠多层，一般会删去水平连接
* 深度RNN结构示意图
<div align="center">
<img src="https://raw.githubusercontent.com/hwt-freedom/AI/master/deep-learning/pictures/RNN/RNN20.jpg" width = "400">
</div>
