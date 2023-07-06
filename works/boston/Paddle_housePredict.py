import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import random

'''
paddle/fluid：飞桨的主库，目前大部分的实用函数均在paddle.fluid包内。
dygraph：动态图的类库。
Linear：神经网络的全连接层函数，即包含所有输入权重相加和激活函数的基本神经元结构。在房价预测任务中，使用只有一层的神经网络（全连接层）来实现线性回归模型。
'''

# 数据处理
def load_data():

    data = np.fromfile('housing.data', sep = ' ')

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD'
                 , 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    data = data.reshape([data.shape[0] // len(feature_names), len(feature_names)])

    ratio = 0.8
    offset = int(data.shape[0]*ratio)
    training_data = data[:offset]

    # 归一化

    maxinums, mininums, avgs = training_data.max(axis=0),training_data.min(axis=0),\
                                training_data.sum(axis=0) / training_data.shape[0]
    global max_values
    global min_values
    global avg_values
    max_values = maxinums
    min_values = mininums
    avg_values = avgs

    for i in range(len(feature_names)):
        data[:,i] = (data[:,i]-avgs[i]) / (maxinums[i] - mininums[i])

    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

# 模型设计

class Regressor(fluid.dygraph.Layer):
    def __init__(self):
        super(Regressor,self).__init__()

        # 定义一层全连接层，输出维度是1，激活函数为None
        self.fc = Linear(input_dim=13, output_dim=1, act=None)

    def forward(self, inputs):
        x = self.fc(inputs)
        return x

# 训练配置
# 定义飞浆动态图的工作环境
with fluid.dygraph.guard(fluid.CUDAPlace(0)):
    # 声明定义好的模型
    model = Regressor()
    # 开启模型训练模式
    model.train()
    # 加载数据
    training_data, test_data = load_data()
    # 定义优化算法，使用SGD ；学习率为 0.01
    opt = fluid.optimizer.SGD(learning_rate=0.01, parameter_list=model.parameters())

with dygraph.guard(fluid.CUDAPlace(0)):
    EPOCH_NUM = 10  # 设置外层循环此时
    BATCH_SIZE = 10 # 设置batch大小

    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        # 在迭代之前打乱每次训练的数据
        np.random.shuffle(training_data)
        # 将训练的数据进行拆分，每个batch包含10条数据
        mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data),BATCH_SIZE)]
        # 定义内层循环
        for iter_id, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:,:-1]).astype('float32')
            y = np.array(mini_batch[:,-1:]).astype('float32')
            # 将numpy数据转化为飞浆的动态图variable
            house_features = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)

            # 前向计算
            predicts = model(house_features)

            # 计算损失
            loss = fluid.layers.square_error_cost(predicts,label=prices)
            avg_loss = fluid.layers.mean(loss)
            if iter_id % 20 == 0:
                print('epoch : {}, iter : {}, loss is : {}'.format(epoch_id, iter_id, avg_loss.numpy()))

            # 反向传播
            avg_loss.backward()
            # 最小化loss，更新参数
            opt.minimize(avg_loss)
            # 清除梯度
            model.clear_gradients()

    # # 保存模型
    # fluid.save_dygraph(model.state_dict(),'LR_model')

with fluid.dygraph.guard():
    pass
#     fluid.save_dygraph(model.state_dict(),'house_predictBypaddle')
#     print('模型保存成功，参数保存在house_predictBypaddle')

