import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split  #划分数据集
from sklearn.compose import  ColumnTransformer  #列转换器
from sklearn.pipeline import Pipeline     #管道操作
from sklearn.impute import SimpleImputer   #缺省值处理
from sklearn.preprocessing import OneHotEncoder,StandardScaler   #独热编码和标准化
from torch.utils.data import TensorDataset,DataLoader  #数据集和数据加载器

from ch06_pytorch.linear_regression import y_pred, epoch_num
print("PyTorch关联的CUDA版本:", torch.version.cuda)
print("CUDA是否可用:", torch.cuda.is_available())

#创建数据集
def create_dataset():
    # 1. 从文件读取数据
    data = pd.read_csv("D:/PythonProject1/data/house_prices.csv")
    # 2. 去除无关列
    data.drop(['Id'], axis=1, inplace=True)
    #3. 划分特征和目标
    X = data.drop("SalePrice",axis=1)
    y = data["SalePrice"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #5.特征工程（特征转换）
    #5.1 按照特征数据类型划分成数值型和类别型
    numerical_features = X.select_dtypes(exclude=['object']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    #5.2定义列转换器
    #5.2.1数值型特征：用平均值填充缺失项，再进行标准化
    numerical_transformer = Pipeline(
        steps=[
            ("fillna", SimpleImputer(strategy="mean")),
            ("std", StandardScaler()),
             ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("fillna", SimpleImputer(strategy="constant",fill_value='NaN')),
            ("Onehot", OneHotEncoder(handle_unknown="ignore")),
             ]
    )
    #5.2.3组合列转换器
    transfomer = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    #5.3进行特征转换，构建新的列，组成最终的数据集
    X_train = transfomer.fit_transform(x_train)
    X_test = transfomer.transform(x_test)
    X_train = pd.DataFrame(X_train.toarray(),columns=transfomer.get_feature_names_out())
    X_test = pd.DataFrame(X_test.toarray(),columns=transfomer.get_feature_names_out())
    #6.构建Tensor数据集
    train_dataset = TensorDataset(torch.tensor(X_train.values,dtype=torch.float32),torch.tensor(y_train.values,dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test.values,dtype=torch.float32),torch.tensor(y_test.values,dtype=torch.float32))
    #返回训练集和测试集，以及特征的数量
    return train_dataset,test_dataset,X_train.shape[1]
#测试主流程
#1.加载数据
train_dataset,test_dataset,feature_num = create_dataset()
#2.构建模型
model = nn.Sequential(
    nn.Linear(feature_num,128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128,1),
)
#3.自定义损失函数
def log_rmse(y_pred,target):
    y_pred = torch.clamp(y_pred,1,float("inf"))
    mse = nn.MSELoss()
    return torch.sqrt(mse(torch.log(y_pred),torch.log(target)))
#4. 模型训练和测试
def train_test(model,train_data,test_data,lr,epoch_num,batch_size,device):
    #1.初始化相关操作
    def init_params(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
    # 1.1参数初始化
    model.apply(init_params)
    #1.2将模型加载到设备
    model = model.to(device)
    #1.3定义一个优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    #定义训练误差和测试误差列表
    train_loss_list = []
    test_loss_list = []
    #2.模型训练
    for epoch in range(epoch_num):
        model.train()
        #
        #2.1创建dataloader
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        train_loss_total = 0
        #2.2按批次迭代训练模型
        for batch_idx,(X,y) in enumerate(train_dataloader):
        #将数据加载到设备
            X,y = X.to(device),y.to(device)
            #2.3.1前向传播
            y_pred = model(X)
            #2.3.2计算损失
            loss_value = log_rmse(y_pred.squeeze(),y)
            #2.3.3反向传播
            loss_value.backward()
            #2.3.4更新参数
            optimizer.step()
            #2.3.5梯度清零
            optimizer.zero_grad()

            #累加损失
            train_loss_total += loss_value.item() * X.shape[0]

        this_train_loss = train_loss_total/len(train_dataset)
        train_loss_list.append(this_train_loss)
        #3.测试
        model.eval()
        #3.1定义dataloader
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
        #3.2计算测试误差
        test_loss_total = 0
        with torch.no_grad():
            for batch_idx,(X,y) in enumerate(test_dataloader):
                X,y = X.to(device),y.to(device)
                y_pred = model(X)
                loss_value = log_rmse(y_pred.squeeze(),y)
                test_loss_total += loss_value.item() * X.shape[0]
                this_test_loss = test_loss_total/len(test_dataset)
        test_loss_list.append(this_test_loss)

        print("epoch:{},train_loss:{},test_loss:{}".format(epoch,this_train_loss,this_test_loss))
    return train_loss_list,test_loss_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.1
epoch_num = 200
batch_size = 64

train_loss_list,test_loss_list = train_test(model,train_dataset,test_dataset,lr,epoch_num,batch_size,device=device)

#画图
plt.plot(train_loss_list,'r-',label="train_loss",linewidth=3)
plt.plot(test_loss_list,'k--',label="test_loss",linewidth=2)
plt.legend()
plt.show()
