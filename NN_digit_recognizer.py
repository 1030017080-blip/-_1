import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from common.load_data import get_data

# 获取数据
x_train, x_test, t_train, t_test = get_data()

#创建模型
net_work = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#设置超参数
learning_rate = 0.1
num_iter = 100
batch_size = 100
num_epochs = 100
train_size = x_train.shape[0]
iter_per_epoch = np.ceil(train_size/batch_size)
iter_num = int(num_iter * iter_per_epoch)
train_loss_list = []
train_acc_list = []
test_acc_list = []
#4.循环迭代，用梯度下降法训练模型
for i in range(iter_num):
    #随机获取批量数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    #4.2计算梯度
    grad = net_work.gradient(x_batch, t_batch)
    #print("grad=====",i)
    #4.3更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        net_work.params[key] -= learning_rate * grad[key] #params 是模型中的参数字典,在前面创建模型时已经初始化过了

    #4.4计算并保存当前的训练损失
    loss = net_work.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    #4.5每完成一个批次，计算并保存训练和测试准确率
    if i % iter_per_epoch == 0:
        train_acc = net_work.accuracy(x_train, t_train)
        test_acc = net_work.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("epoch:{},loss:{},Train_acc:{},Test_acc:{}".format(i//iter_per_epoch + 1,loss,train_acc,test_acc)) #第一个args用if条件的值来求

#画图
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.legend (loc='best')
plt.show()