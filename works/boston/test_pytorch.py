import torch
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
 
# def test():   
#     # test
#     x_data = torch.tensor(tp.X_test, dtype=torch.float32)
#     y_data = torch.tensor(tp.Y_test, dtype=torch.float32)
#     x_data,y_data =x_data.to(tp.device),y_data.to(tp.device)
#     pred = tp.net.forward(x_data)
#     pred = torch.squeeze(pred)
#     loss_test = tp.loss_func(pred, y_data) * 0.001
#     print("ite:{}, loss_test:{}".format(1, loss_test))
