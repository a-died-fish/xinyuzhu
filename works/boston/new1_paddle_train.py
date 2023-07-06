# 导入基本的库
import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib.pyplot as plt

BUF_SIZE=500
BATCH_SIZE=20

#用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.train(), 
                          buf_size=BUF_SIZE),                    
    batch_size=BATCH_SIZE)   
#用于测试的数据提供器，每次从缓存中随机读取批次大小的数据
test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.test(),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)  

#用于打印，查看uci_housing数据
train_data=paddle.dataset.uci_housing.train()
sampledata=next(train_data())
print(sampledata)
'''
(array([-0.0405441 ,  0.06636364, -0.32356227, -0.06916996, -0.03435197,
        0.05563625, -0.03475696,  0.02682186, -0.37171335, -0.21419304,
       -0.33569506,  0.10143217, -0.21172912]), array([24.]))
'''
paddle.enable_static()
#定义张量变量x，表示13维的特征值
#x = paddle.static.data(name='x', shape=[13], dtype='float32')
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
#定义张量y,表示目标值
#y = paddle.static.data(name='y', shape=[1], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
#定义一个简单的线性网络,连接输入和输出的全连接层
#input:输入tensor;
#size:该层输出单元的数目
#act:激活函数
y_predict=fluid.layers.fc(input=x,size=1,act=None)

cost = fluid.layers.square_error_cost(input=y_predict, label=y) #求一个batch的损失值
avg_cost = fluid.layers.mean(cost)                              #对损失值求平均值

optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)
test_program = fluid.default_main_program().clone(for_test=True)

use_cuda = True                         #use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU           
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)              #创建一个Executor实例exe
exe.run(fluid.default_startup_program()) #Executor的run()方法执行startup_program(),进行参数初始化

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])#feed_list:向模型输入的变量表或变量表名

iter=0
iters=[]
train_costs=[]

def draw_train_process(iters,train_costs):
     title="training cost"
     plt.title(title, fontsize=24)
     plt.xlabel("iter", fontsize=14)
     plt.ylabel("cost", fontsize=14)
     plt.plot(iters, train_costs,color='red',label='training cost') 
     plt.grid()
     plt.savefig("training_testing_loss.png",dpi=300)
     plt.show()


EPOCH_NUM=50
model_save_dir = "model/new1_paddle.model"

for pass_id in range(EPOCH_NUM):                                  #训练EPOCH_NUM轮
    # 开始训练并输出最后一个batch的损失值
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):              #遍历train_reader迭代器
        train_cost = exe.run(program=fluid.default_main_program(),#运行主程序
                             feed=feeder.feed(data),              #喂入一个batch的训练数据，根据feed_list和data提供的信息，将输入数据转成一种特殊的数据结构
                             fetch_list=[avg_cost])    
        if batch_id % 40 == 0:
            print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))    #打印最后一个batch的损失值
            print(avg_cost)
        iter=iter+BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0][0])
       
   
    # 开始测试并输出最后一个batch的损失值
    test_cost = 0
    for batch_id, data in enumerate(test_reader()):               #遍历test_reader迭代器
        test_cost= exe.run(program=test_program, #运行测试cheng
                            feed=feeder.feed(data),               #喂入一个batch的测试数据
                            fetch_list=[avg_cost])                #fetch均方误差
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost[0][0]))     #打印最后一个batch的损失值
    
    #保存模型
    # 如果保存路径不存在就创建


# if not os.path.exists(model_save_dir):
#     os.makedirs(model_save_dir)
# print ('save models to %s' % (model_save_dir))
# #保存训练参数到指定路径中，构建一个专门用预测的program
# fluid.io.save_inference_model(model_save_dir,   #保存推理model的路径
#                                   ['x'],            #推理（inference）需要 feed 的数据
#                                   [y_predict],      #保存推理（inference）结果的 Variables
#                                   exe)              #exe 保存 inference model
draw_train_process(iters,train_costs)


