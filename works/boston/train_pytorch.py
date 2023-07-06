#使用pytorch框架 进行gpu并行运算


import torch
# data
import numpy as np
import re
 
# 读取数据文件
ff = open("housing.data").readlines()
data = []
# 对每一项进行解析
for item in ff:
    # 数据通过空格分割，多个空格合并成一个空格
    out = re.sub(r"\s{2,}", " ", item).strip()
    print(out)
    data.append(out.split(" "))
 
# 转换成float
data = np.array(data).astype(np.float64)
print(data.shape)
# (506,14) 506条数据，14个特征。前3个是x后面是y
# 再对数据进行切分
Y = data[:, -1]
X = data[:, 0:-1]
# 训练集：前496个样本
X_train = X[0:496, ...]
Y_train = Y[0:496, ...]
# 测试集：剩下的样本
X_test = X[496:, ...]
Y_test = Y[496:, ...]
 
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
 
 
# net
class Net(torch.nn.Module):
    # 初始化，传入feature和输出
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        # 隐藏层
        self.hidden = torch.nn.Linear(n_feature, 100)
        # 线性回归
        self.predict = torch.nn.Linear(100, n_output)
 
    # x是输入
    def forward(self, x):
        # 调用隐藏层
        out = self.hidden(x)
        # 对输出加入relu非线性运算
        out = torch.relu(out)
        # 输出预测
        out = self.predict(out)
        return out
 
 
# 初始化网络，特征数量13，输出特征数量1个
net = Net(13, 1)
# loss
loss_func = torch.nn.MSELoss()
# optimiter优化器，传入参和学习率，0.01
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

#gpu计算
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
net = torch.nn.DataParallel(net)
net.to(device)

def train(epoch):
    # training，1000次
    for i in range(epoch):
        # 训练集初始化
        x_data = torch.tensor(X_train, dtype=torch.float32)
        y_data = torch.tensor(Y_train, dtype=torch.float32)
        x_data,y_data =x_data.to(device),y_data.to(device)
        # 前向运算，根据x预测y
        pred = net.forward(x_data)
        pred = torch.squeeze(pred)
        # 计算loss
        loss = loss_func(pred, y_data) * 0.001
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印迭代次数和loss的变化
        print("ite:{}, loss_train:{}".format(i, loss))
        # 预测结果的前10个值
        print(pred[0:10])
        # 真实结果的前10个值
        print(y_data[0:10])

def test():   
    # test
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)
    x_data,y_data =x_data.to(device),y_data.to(device)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 0.001
    print("ite:{}, loss_test:{}".format(1, loss_test))
 
print('cuda number',torch.cuda.device_count())
train(100)
test()


#torch.save(net, "model/model.pkl")
# torch.load("")
# torch.save(net.state_dict(), "params.pkl")
# net.load_state_dict("")