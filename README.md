# Deep-learning-items
Coursera 深度学习

# 第一章

1 Basics With Numpy v3   

基础函数（sigmoid，softmax） 向量化（np.dot multiply outer）

 

2 Logistic Regression with a Neural Network mindset v5 

Numpy实现 基本网络（无隐层）的结构  initialize(), propagate(), optimize()， model().

 

3 Planar data classification with one hidden layer v5

Numpy实现 一个隐层的NN的2维数据分类器的构建过程    forward 4 backward 6

 

4 Building your Deep Neural Network Step by Step v8

*[LINEAR->RELU]* × *(L-1)  ->  LINEAR ->  SIGMOID*

 

5 Deep Neural Network - Application v8

自己搭N隐层网络   

 

# 第二章

1 Initialization

随机参数初始化 0 randn /sqrt(2/layers)

 

2 Regularization - v2

正则化和dropout

 

3 Gradient+Checking v1

梯度检验

 

4 Optimization+methods

gd momentum adam 算法更新参数

 

5 Tensorflow+Tutorial

tf基本教程会话等，tf搭3层网络

 

# 第四章

1 Convolutional Model: step by step

Numpy实现 卷积层(padding, 滑动窗口的实现)，池化层(maxpooling)

 

2 Convolutional Model: application

Tensorflow实现 convnet

 

3 Keras+-+Tutorial+-+Happy+House+v2  

\1.     keras的基本操作 Create the model by calling the function above

\2.     Compile the model by calling model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])

\3.     Train the model on train data by calling model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)

\4.     Test the model on test data by calling model.evaluate(x = ..., y = ...)

实现简单的cnn网络

 

4 ResNet50

Keras 实现 非常深的“普通”网络在实践中不起作用。

跳跃连接有助于解决消失梯度问题，使ResNet块更容易学习恒等函数。

主要有两种类型的块:单位块和卷积块。

 

5 Car detection with YOLOv2

Input image (608, 608, 3)

The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.

After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):

​    Each cell in a 19x19 grid over the input image gives 425 numbers.

​    425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.

​    85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw)(pc,bx,by,bh,bw)  has 5 numbers, and and 80 is the number of classes we'd like to detect

You then select only few boxes based on:

​    Score-thresholding: throw away boxes that have detected a class with a score less than the threshold

​    Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes

This gives you YOLO's final output.

 

6 Art generation with Neural Style Transfer

Tensorflow实现风格迁移 代价函数的定义

 

7 Face Recognition for the Happy House

Keras 实现Facenet (inception net的实现)，人脸识别的triplet loss
 
 

 

# 第五章

1 Building a recurrent neural network - step by step

Numpy实现rnn网络，lstm和gru等cell

 

2 Dinosaur Island - Character-Level Language Modeling

Numpy实现语言模型(梯度clip)

Keras实现多层lstm 莎士比亚text generation

https://github.com/keras-team/keras/tree/master/examples

 

3 Jazz improvisation with LSTM

Keras实现多层lstm  sequence generation

 

4 Operations on word vectors – Debiasing

加载预先训练好的单词向量，并使用余弦相似性度量相似性

使用词的嵌入来解决词的类比问题，如男人对女人就像国王对____一样。

修改单词嵌入以减少性别偏见

 

5 Emojify

情感分析

 

6 Neural Machine Translation with Attention

keras机器翻译添加注意力机制

 

7 Trigger word detection

keras触发词检测 非常深度的RNN网络

![img](file:///C:/Users/zzz/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)