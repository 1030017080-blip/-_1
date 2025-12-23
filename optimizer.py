import numpy as np

# 随机梯度下降 SGD
class SGD:
    # 初始化
    def __init__(self, lr=0.01):
        self.lr = lr
    # 参数更新，传入参数字典和梯度字典
    def update(self, params, grads):
        # 遍历传入的所有参数，按照公式更新
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# 动量法
class Momentum:
    # 初始化
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None   # 历史负梯度的加权和
    # 更新方法
    def update(self, params, grads):
        # 对v进行初始化
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        # 按照公式进行参数更新
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

# AdaGrad
class AdaGrad:
    # 初始化
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None \
    # 更新方法
    def update(self, params, grads):
        # 对h进行初始化
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        # 按照公式进行参数更新
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)

# RMSProp
class RMSProp:
    # 初始化
    def __init__(self, lr=0.01, decay=0.99):
        self.lr = lr
        self.decay = decay
        self.h = None
    # 更新方法
    def update(self, params, grads):
        # 对h进行初始化
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        # 按照公式进行参数更新
        for key in params.keys():
            self.h[key] *= self.decay
            self.h[key] += (1 - self.decay) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)

# Adam
class Adam:
    # 初始化
    def __init__(self, lr=0.01, alpha1=0.9, alpha2=0.999):
        self.lr = lr
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.v = None
        self.h = None
        self.t = 0  # 迭代次数
    # 更新方法
    def update(self, params, grads):
        # 初始化v和h
        if self.v is None:
            self.v, self.h = {}, {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        self.t += 1 # 迭代次数+1
        # 按照当前的迭代次数，改变学习率参数
        lr_t = self.lr * np.sqrt(1 - self.alpha2**self.t) / (1 - self.alpha1**self.t)
        # 遍历所有参数，按公式进行更新
        for key in params.keys():
            # self.v[key] = self.alpha1 * self.v[key] + (1 - self.alpha1) * grads[key]
            # self.h[key] = self.alpha2 * self.h[key] + (1 - self.alpha2) * (grads[key] ** 2)
            self.v[key] += (1 - self.alpha1) * (grads[key] - self.v[key])
            self.h[key] += (1 - self.alpha2) * (grads[key] ** 2 - self.h[key])
            params[key] -= lr_t * self.v[key] / (np.sqrt(self.h[key]) + 1e-8)