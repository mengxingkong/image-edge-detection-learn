###  大牛 [David](https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/03_numpy_neural_net/Numpy%20deep%20neural%20network.ipynb) 【numpy 实现简单神经网络】 代码理解

准备着手学习图像处理以及神经网络的内容，但是感觉自己对神经网络的基础了解的太少，从来没有手写过简单的神经网络，而且对于“神奇”的反向传播算法一直是云里雾里的，所以准备抄写一下别人的代码，实现一遍神经网络和反向传播算法，大牛的github代码给了我很大的帮助，但是代码中还是有一些地方绕了很久，在这里做一个记录。

具体的神经网络模型大家可以去大牛的github上面去看

1. 前向传播算法中的 Z_curr 的 计算：

    ```angular2html
     Z_curr = np.dot(W_curr, A_prev) + b_curr
    ```
    
    在我的理解中 神经网络的点乘运算应该是：
    
    ```angular2html
    输入值 * 权重 + 偏置 = 输出值
    ``` 
    
    和大牛的代码中有冲突，这里大牛这样计算是为了满足这个神经网络的输出，大牛代码中神经网络的输出是 （1，900） 的格式， 所以神经网络的计算过程是：
    
    ```angular2html
    W1[25,2] * A0[2,900]
    W2[50,25] * A1[25,900]
    ...
    W5[1,25] * A[4][25,900]
    
    ```
    
    和这里计算方式对应的就是在传入训练参数时，进行的数组的reshape，和数组的转至操作

2. 反向传播中导数的计算 （重点）

    首先是 dZ_curr = backward_activation_func(dA_curr, Z_curr)，这里我们假设其中的一层神经网络的连接是这样的：
    
    ```angular2html
    画图 有点难，先写到纸上整理吧
    ```