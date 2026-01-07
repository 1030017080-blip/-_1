import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#导入torch的数据源 数据迭代器工具包
from torch.utils.data import DataLoader,Dataset
#用于获得常见字母及字符规范化
import string
#导入时间工具包’
import time
from tqdm import tqdm
#导入制图工具包
import matplotlib.pyplot as plt

#获取常见的字符
letters = string.ascii_letters + " ,.;'"
n_letters = len(letters)
# print(n_letters)  每完成一个操作就打印测试一下
# print(letters)
#国家名 种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
n_categorys = len(categorys)
# print(n_categorys)
# # print(categorys)

#读取数据到内存
def read_data(filepath):
    #定义两个空列表，一个存储x，一个存储y
    my_list_x,my_list_y = [],[]
    with open(filepath,mode='r',encoding='utf-8')as f:
        lines = f.readlines()

    for line in lines:
        if len(line)<=5:
            continue #txt文档里一般字符长度都大于5，这个就是处理异常值
        x,y = line.strip().split('\t')
        my_list_x.append(x)
        my_list_y.append(y)
    return my_list_x,my_list_y
#构建Dataset
class NameClassDataset(Dataset):
    def __init__(self,my_list_x,my_list_y):
        super().__init__()
        #样本x
        self.my_list_x = my_list_x
        #标签y
        self.my_list_y = my_list_y
        #获取样本长度
        self.sample_len = len(my_list_x)
#定义魔法方法
#是为了让你的NameClassDataset类能像 “容器” 一样方便使用（比如像列表那样用len()和索引取值）
    def __len__(self):
        return self.sample_len
    #定义getitem方法（魔法方法）
    def __getitem__(self, index):
        #index防止索引溢出
        index = min(max(0,index),self.sample_len-1)#当索引为负取0，为大于最大索引取最大索引
        #根据index取出对应的样本和标签
        x = self.my_list_x[index]
        y = self.my_list_y[index]
        # print(f'x-->{x}')
        # print(f'y-->{y}')

        #将x转换为one-hot编码张量形式
        tensor_x = torch.zeros(len(x),n_letters)
        # print(f'tensor_x-->{tensor_x}')
        for ix,letter in enumerate(x):
            tensor_x[ix][letters.find(letter)] = 1
        # print(f'tensor_x-->{tensor_x}') 该循环是做一个one-hot编码x的张量的构建，名字有n个字母进行n次循环，最后是n*57的一个one-hot编码张量
        tensor_y = torch.tensor(categorys.index(y),dtype=torch.long)
#把类别y映射为它在类别列表categories中的索引
#（比如"czech"是categories里的第 2 个，索引就是 2），最终得到一个标量张量（代表样本的类别标签）。
        return tensor_x,tensor_y
#测试一下dataset
def test_dataset():
    my_list_x,my_list_y = read_data('./name_classfication.txt')
    #构建dataset
    my_dataset = NameClassDataset(my_list_x,my_list_y)
    print( len( my_dataset)) #同样 print(my_dataset.__len())也可以获取数据集长度
    tensor_x,tensor_y = my_dataset[0]
    #也可以这样写
    #tensor_y = my_dataset.__getitem__(0)
    # print(f'tensor_x-->{tensor_x}')
    # print(f'tensor_y-->{tensor_y}')


#定义迭代器 对dataset进行再次封装
def get_tataloader():
    my_list_x,my_list_y = read_data('./name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x,my_list_y)
    #my_dataloader的作用是批量获取数据，每次获取batch_size个数据，并返回一个迭代器。
    my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)#每个人名字长度不同，一次只获取一个，不然不能统一正确处理
    for tensor_x,tensor_y in my_dataloader:
        print(f'tensor_x-->{tensor_x.shape}')
        print(f'tensor_y-->{tensor_y}')
        break

#定义RNN模型
class My_RNN(nn.Module): #继承nn.Module基类,搭建自己的神经网络模型,固定用法
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        #input_size:输入向量的维度
        self.input_size = input_size
        #hidden_size:隐藏层的维度
        self.hidden_size = hidden_size
        #output_size:输出向量的维度 18
        self.output_size = output_size
        #num_layers:隐藏层的层数
        self.num_layers = num_layers
        #定义RNN层
        self.rnn = nn.RNN(input_size,hidden_size,num_layers)
        #定义全连接层
        self.linear = nn.Linear(self.hidden_size,self.output_size)
        #定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)


    #前向传播
    def forward(self,input,hidden):
        #input输入是二维张量，[n,57]n个单词，每个单词用57个数字表示(one-hot编码)
        #rnn模型接受三维，因此需要升维
        input = input.unsqueeze(1)
        #将input和hidden送入模型
        output,hn = self.rnn(input,hidden)
        # print(f'output-->{output.shape}')
        # print(f'hn-->{hn.shape}')
        result = self.linear(output[-1])
        # print(f'result-->{result.shape}')
        return self.softmax(result),hn


    #初始化hidden
    def inithidden(self):
        return torch.zeros(self.num_layers,1,self.hidden_size)




#定义lstm模型
class My_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        #定义lstm层
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.num_layers)
        #定义全连接层
        self.linear = nn.Linear(self.hidden_size,self.output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self,input,hidden,c):
        input = input.unsqueeze(1)
        output,(hn,cn) = self.lstm(input,(hidden,c))
        #取出最后一个单词的隐藏层张量表示
        temp = output[-1]
        result = self.linear(temp)
        return self.softmax(result),hn,cn

    def inithidden(self):
        #return torch.zeros(self.num_layers,1,self.hidden_size),torch.zeros(self.num_layers,1,self.hidden_size)
        h0 = torch.zeros(self.num_layers,1,self.hidden_size)
        c0 = torch.zeros(self.num_layers,1,self.hidden_size)
        return h0,c0


#定义GRU模型
class My_GRU(nn.Module): #继承nn.Module基类,搭建自己的神经网络模型,固定用法
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        #input_size:输入向量的维度
        self.input_size = input_size
        #hidden_size:隐藏层的维度
        self.hidden_size = hidden_size
        #output_size:输出向量的维度 18
        self.output_size = output_size
        #num_layers:隐藏层的层数
        self.num_layers = num_layers
        #定义RNN层
        self.gru = nn.GRU(input_size,hidden_size,num_layers)
        #定义全连接层
        self.linear = nn.Linear(self.hidden_size,self.output_size)
        #定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)


    #前向传播
    def forward(self,input,hidden):
        #input输入是二维张量，[n,57]n个单词，每个单词用57个数字表示(one-hot编码)
        #rnn模型接受三维，因此需要升维
        input = input.unsqueeze(1)
        #将input和hidden送入模型
        output,hn = self.gru(input,hidden)
        # print(f'output-->{output.shape}')
        # print(f'hn-->{hn.shape}')
        result = self.linear(output[-1])
        # print(f'result-->{result.shape}')
        return self.softmax(result),hn


    #初始化hidden
    def inithidden(self):
        return torch.zeros(self.num_layers,1,self.hidden_size)





#测试模型
def test_model():
    #实例化
    my_rnn = My_RNN(input_size=57,hidden_size=128,output_size=18)
    my_lstm = My_LSTM(input_size=57,hidden_size=128,output_size=18)
    my_gru = My_GRU(input_size=57,hidden_size=128,output_size=18)
    my_list_x,my_list_y = read_data('./name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x,my_list_y)
    #my_dataloader的作用是批量获取数据，每次获取batch_size个数据，并返回一个迭代器。
    my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)#每个人名字长度不同，一次只获取一个，不然不能统一正确处理
    for tensor_x,tensor_y in my_dataloader:
        #从my_dataloader里面迭代出来的tensor_x shape：[1,5,57] batch_size,seq_len,input_size
        input = tensor_x[0]
        hidden = my_gru.inithidden()
        output,hn = my_gru(input,hidden)
        print(f'output-->{output.shape}')
        # print(f'hn-->{hn.shape}')
        break



#定义全局变量参数
my_lr = 1e-3
epochs = 1

#训练rnn模型
def train_rnn():
    #读取数据
    my_list_x,my_list_y = read_data('./name_classfication.txt')
    #实例化dataset数据源对象
    my_dataset = NameClassDataset(my_list_x,my_list_y)
    #实例化模型
    #n_letters = 57,hidden_size=128,output_size=18
    my_rnn = My_RNN(input_size=57,hidden_size=128,output_size=18)
    #实例化损失函数对象
    my_nll_loss = nn.NLLLoss ()
    #实例化优化器对象
    my_optim = optim.Adam(my_rnn.parameters(),lr=my_lr)
    #定义打印日志的参数、
    start_time = time.time()
    total_iter_num = 0
    total_loss = 0
    total_loss_list = []
    total_acc_num = 0
    total_acc_list = []
    #开始训练
    for epoch in range(epochs):
        #实例化dataloader
        my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)
        for i ,(x,y) in enumerate(tqdm(my_dataloader)):
            # print(f'x--》{x.shape}')
            # print(f'y--》{y}')
            output, hn = my_rnn(input=x[0], hidden=my_rnn.inithidden())
            # print(f'output--》{output}') # [1, 18]
            #计算损失
            my_loss = my_nll_loss(output,y)
            # print(f'my_loss--》{my_loss}')
            #梯度清零
            my_optim.zero_grad()
            #方向传播
            my_loss.backward()
            #更新参数
            my_optim.step()
            #统计一下已经训练样本的总个数
            total_iter_num += 1
            #统计一下已经训练样本的总损失
            total_loss += my_loss.item()

            i_predict_num = 1 if torch.argmax(output).item() == y.item() else 0
            #三元运算，argmax取的是预测值最大的索引
            total_acc_num = total_acc_num + i_predict_num
            #每隔100次训练保存一下平均损失和准确率
            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss / total_iter_num)
                total_acc_list.append(total_acc_num / total_iter_num)
            #每隔2000次训练打印日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch+1, temp_loss, temp_time, temp_acc))

    torch.save(my_rnn.state_dict(),'./save_model/rnn_%d.bin'%(epoch+1))
    #计算总时间
    total_time = int(time.time() - start_time)
    #将结果保存到文件中
    dict1 = {"avg_loss": total_loss_list,
        "avg_acc":total_acc_list,
             "total_time":total_time
    }
    with open('./save_results/rnn.json','w') as fw:
        json.dump(dict1, fw)  # 直接把dict1写入fw这个文件对象
    return total_loss_list,total_acc_list,total_time

#训练lstm模型
def train_lstm():
    #读取数据
    my_list_x,my_list_y = read_data('./name_classfication.txt')
    #实例化dataset数据源对象
    my_dataset = NameClassDataset(my_list_x,my_list_y)
    #实例化模型
    #n_letters = 57,hidden_size=128,output_size=18
    my_lstm = My_LSTM(input_size=57,hidden_size=128,output_size=18)
    #实例化损失函数对象
    my_nll_loss = nn.NLLLoss ()
    #实例化优化器对象
    my_optim = optim.Adam(my_lstm.parameters(),lr=my_lr)
    #定义打印日志的参数、
    start_time = time.time()
    total_iter_num = 0
    total_loss = 0
    total_loss_list = []
    total_acc_num = 0
    total_acc_list = []
    #开始训练
    for epoch in range(epochs):
        #实例化dataloader
        my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)
        for i ,(x,y) in enumerate(tqdm(my_dataloader)):
            # print(f'x--》{x.shape}')
            # print(f'y--》{y}')
            hidden,cn = my_lstm.inithidden()
            output, hn ,cn= my_lstm(input=x[0], hidden=hidden,c=cn)
            # print(f'output--》{output}') # [1, 18]
            #计算损失
            my_loss = my_nll_loss(output,y)
            # print(f'my_loss--》{my_loss}')
            #梯度清零
            my_optim.zero_grad()
            #方向传播
            my_loss.backward()
            #更新参数
            my_optim.step()
            #统计一下已经训练样本的总个数
            total_iter_num += 1
            #统计一下已经训练样本的总损失
            total_loss += my_loss.item()

            i_predict_num = 1 if torch.argmax(output).item() == y.item() else 0
            #三元运算，argmax取的是预测值最大的索引
            total_acc_num = total_acc_num + i_predict_num
            #每隔100次训练保存一下平均损失和准确率
            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss / total_iter_num)
                total_acc_list.append(total_acc_num / total_iter_num)
            #每隔2000次训练打印日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch+1, temp_loss, temp_time, temp_acc))

    torch.save(my_lstm.state_dict(),'./save_model/lstm_%d.bin'%(epoch+1))
    #计算总时间
    total_time = int(time.time() - start_time)
    #将结果保存到文件中
    dict2 = {"avg_loss":total_loss_list,
        "avg_acc":total_acc_list,
             "total_time":total_time
    }
    with open('./save_results/lstm.json','w') as fw:
        json.dump(dict2, fw)  # 直接把dict2写入fw这个文件对象
    return total_loss_list,total_acc_list,total_time

#训练gru模型
def train_gru():
    #读取数据
    my_list_x,my_list_y = read_data('./name_classfication.txt')
    #实例化dataset数据源对象
    my_dataset = NameClassDataset(my_list_x,my_list_y)
    #实例化模型
    #n_letters = 57,hidden_size=128,output_size=18
    my_gru = My_GRU(input_size=57,hidden_size=128,output_size=18)
    #实例化损失函数对象
    my_nll_loss = nn.NLLLoss ()
    #实例化优化器对象
    my_optim = optim.Adam(my_gru.parameters(),lr=my_lr)
    #定义打印日志的参数、
    start_time = time.time()
    total_iter_num = 0
    total_loss = 0
    total_loss_list = []
    total_acc_num = 0
    total_acc_list = []
    #开始训练
    for epoch in range(epochs):
        #实例化dataloader
        my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)
        for i ,(x,y) in enumerate(tqdm(my_dataloader)):
            # print(f'x--》{x.shape}')
            # print(f'y--》{y}')
            output, hn = my_gru(input=x[0], hidden=my_gru.inithidden())
            # print(f'output--》{output}') # [1, 18]
            #计算损失
            my_loss = my_nll_loss(output,y)
            # print(f'my_loss--》{my_loss}')
            #梯度清零
            my_optim.zero_grad()
            #方向传播
            my_loss.backward()
            #更新参数
            my_optim.step()
            #统计一下已经训练样本的总个数
            total_iter_num += 1
            #统计一下已经训练样本的总损失
            total_loss += my_loss.item()

            i_predict_num = 1 if torch.argmax(output).item() == y.item() else 0
            #三元运算，argmax取的是预测值最大的索引
            total_acc_num = total_acc_num + i_predict_num
            #每隔100次训练保存一下平均损失和准确率
            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss / total_iter_num)
                total_acc_list.append(total_acc_num / total_iter_num)
            #每隔2000次训练打印日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch+1, temp_loss, temp_time, temp_acc))

    torch.save(my_gru.state_dict(),'./save_model/gru_%d.bin'%(epoch+1))
    #计算总时间
    total_time = int(time.time() - start_time)
    #将结果保存到文件中
    dict3 = {"avg_loss":total_loss_list,
        "avg_acc":total_acc_list,
             "total_time":total_time
    }
    with open('./save_results/gru.json','w') as fw:
        json.dump(dict3, fw)  # 直接把dict3写入fw这个文件对象

def read_json(data_path):
    with open(data_path,'r') as fr:
        rnn_results = json.loads(fr.read())
    avg_loss = rnn_results["avg_loss"]
    total_time = rnn_results["total_time"]
    avg_acc = rnn_results["avg_acc"]
    return avg_loss, total_time, avg_acc

#绘制图像
def dm_show_results():
    rnn_avg_loss, rnn_total_time, rnn_avg_acc = read_json('./save_results/rnn.json')
    lstm_avg_loss, lstm_total_time, lstm_avg_acc = read_json('./save_results/lstm.json')
    gru_avg_loss, gru_total_time, gru_avg_acc = read_json('./save_results/gru.json')
    #对比不同模型的损失
    plt.figure(0)
    plt.plot(rnn_avg_loss, label='rnn')
    plt.plot(lstm_avg_loss, label='lstm',color='r')
    plt.plot(gru_avg_loss, label='gru',color='g')
    plt.legend(loc='upper left')
    plt.savefig('./img/loss.png')
    plt.show()
    #对比不同模型准确率
    plt.figure(1)
    plt.plot(rnn_avg_acc, label='rnn')
    plt.plot(lstm_avg_acc, label='lstm',color='r')
    plt.plot(gru_avg_acc, label='gru',color='g')
    plt.legend(loc='upper left')
    plt.savefig('./img/acc.png')
    plt.show()
    #对比不同模型耗时
    plt.figure(2)
    x_data = ["rnn","lstm","gru"]
    y_data = [rnn_total_time,lstm_total_time,gru_total_time]
    plt.bar(range(len(x_data)),y_data,tick_label=x_data)
    plt.legend(loc='upper left')
    plt.savefig('./img/time.png')
    plt.show()


#构造将字符串（人名)转换为张量的函数
def line2tensor(x):
    tensor_x = torch.zeros(len(x),n_letters)
    for i,letter in enumerate(x):
        tensor_x[i][letters.find(letter)] = 1
    return tensor_x
#构造rnn预测函数
def rnn_predict(x):
    #将数据x进行张量的转换
    tensor_x = line2tensor(x)
    #加载训练好的参数
    my_rnn = My_RNN(input_size=57,hidden_size=128,output_size=18)
    my_rnn.load_state_dict(torch.load('./save_model/rnn_1.bin'))
    #实现模型的预测
    with torch.no_grad():
        #将数据送入模型进行预测
        output,hn = my_rnn(input=tensor_x,hidden=my_rnn.inithidden())
        # print( output)
        #获取预测结果最大的前三个值
        values,indexes = torch.topk(output,3)
        print(values)
        print(indexes)
        for i in range(3):
            value = values[0][i]
            index = indexes[0][i]
            category = categorys[index]
            print('预测的概率是%.3f,预测的类别是%s' %(value,category))

#构造lstm预测函数
def lstm_predict(x):
    # 将数据x进行张量的转换
    tensor_x =  line2tensor(x)
    # 加载训练好的模型
    my_lstm = My_LSTM(input_size=57, hidden_size=128, output_size=18)
    my_lstm.load_state_dict(torch.load('./save_model/lstm_1.bin'))
    # 实现模型的预测
    with torch.no_grad():
        # 将数据送入模型
        hidden, c = my_lstm.inithidden()
        output, hn, cn = my_lstm(tensor_x, hidden, c)
        # print(f'output--》{output}')
        # 获取output最大的前3个值
        # output.topk(3, 1, True)
        values, indexes = torch.topk(output, k=3, dim=-1, largest=True)
        print(f'values-->{values}')
        print(f'indexes-->{indexes}')
        for i in range(3):
            value = values[0][i]
            index = indexes[0][i]
            category = categorys[index]
            print(f'当前预测的值是：{value}, 国家类别是：{category}')


#构造gru预测函数
def gru_predict(x):
    # 将数据x进行张量的转换
    tensor_x =  line2tensor(x)
    # 加载训练好的模型
    my_gru = My_GRU(input_size=57, hidden_size=128, output_size=18)
    my_gru.load_state_dict(torch.load('./save_model/gru_1.bin'))
    # 实现模型的预测
    with torch.no_grad():
        # 将数据送入模型
        output,hn = my_gru(tensor_x,my_gru.inithidden())
        # print(f'output--》{output}')
        # 获取output最大的前3个值
        # output.topk(3, 1, True)
        values, indexes = torch.topk(output, k=3, dim=-1, largest=True)
        print(f'values-->{values}')
        print(f'indexes-->{indexes}')
        for i in range(3):
            value = values[0][i] #返回前三大预测值
            index = indexes[0][i] #取第i个预测值的类别下表
            category = categorys[index] #取出对应的类别
            print(f'当前预测的值是：{value}, 国家类别是：{category}')
if __name__ == '__main__':
    #test_dataset()
   #get_tataloader()
    # test_model()
    # train_rnn()
    # train_lstm()
    # train_gru()
    # dm_show_results()
    rnn_predict(x="zhang")
    print('*'*80)
    lstm_predict(x="zhang")
    print('*' * 80)
    gru_predict(x="zhang")
