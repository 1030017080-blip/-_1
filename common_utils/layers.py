from common.functions import *

# ReLU
class Relu:
    # 初始化
    def __init__(self):
        # 内部属性，记录哪些x<=0
        self.mask = None
    # 前向传播
    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        # 将x<=0的值都赋为0
        y[self.mask] = 0
        return y
    # 反向传播
    def backward(self, dy):
        dx = dy.copy()
        # 将x<=0的值都赋为0
        dx[self.mask] = 0
        return dx

# Sigmoid
class Sigmoid:
    # 初始化
    def __init__(self):
        # 定义内部属性，记录输出值y，用于反向传播时计算梯度
        self.y = None
    # 前向传播
    def forward(self, x):
        y = sigmoid(x)
        self.y = y
        return y
    # 反向传播
    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y
        return dx

# Affine 仿射层
class Affine:
    # 初始化
    def __init__(self, W, b):
        self.W = W
        self.b = b
        # 对输入数据X做保存，方便反向传播计算梯度
        self.X = None
        self.original_x_shape = None
        # 将权重和偏置参数的梯度（偏导数）保存成属性，方便梯度下降法计算
        self.dW = None
        self.db = None
    # 前向传播
    def forward(self, X):
        self.original_x_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        y = np.dot(self.X, self.W) + self.b
        return y
    # 反向传播
    def backward(self, dy):
        dX = np.dot(dy, self.W.T)
        dX = dX.reshape(*self.original_x_shape)
        self.dW = np.dot(self.X.T, dy)
        self.db = np.sum(dy, axis=0)
        return dX

# 输出层
class SoftmaxWithLoss:
    # 初始化
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    # 前向传播
    def forward(self, X, t):
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy(self.y, self.t)
        return self.loss
    # 反向传播
    def backward(self, dy=1):
        n = self.t.shape[0]
        # 如果是独热编码的标签，就直接代入公式计算
        if self.t.size == self.y.size:
            dx = self.y - self.t
        # 如果是顺序编码的标签，就需要找到分类号对应的值，然后相减
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1
        return dx / n

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y =None
        self.t =None
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y,self.t)
        return self.loss

    def backforward(self,dout =1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size),self.t] -= 1
            return dx / batch_size
        return dx
