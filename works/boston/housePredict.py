import numpy as np
import json
import matplotlib.pyplot as plt

# 读入数据


def load_data():

    datafile = np.fromfile('housing.data', sep=' ')
    # print(datafile)

    # 数据结构变换

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD'
                 , 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    ''' 
    特征名：
    CRIM：该镇的人均犯罪率，ZN：占地面积超过25000平方米的住宅用地比例
    INDS:非零售商业用地比例 CHAS：是否临近河流 NOX：一氧化氮浓度
    RM： 每栋房屋的平均客房数 AGE：1940年之前建成的自用单位比例
    DIS：到波士顿5个就业中心的加权距离 RAD：到径向公路的可达性指数
    TAX：全值财产税率 PTRATIO：学生与教师的比例 B: 1000（BK-0.63）^2
    LSTAT：低收入人群比例  MEDV：同类房屋的价格中位数
    '''

    feature_num = len(feature_names)
    data = datafile.reshape([datafile.shape[0]//feature_num, feature_num])

    # 检查
    # first_element = data[0]
    # print(first_element)
    # print(first_element.shape)

    # 划分训练集和验证集

    ratio = 0.8
    offset = int(data.shape[0]*ratio)
    training_data = data[:offset]
    # print(training_data.shape)


    '''
    对特征的取值进行归一化处理，使得每个特征的取值缩放到0~1之间。
    这样做有两个好处：一是模型训练更高效；
    二是特征前的权重大小可以代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）
    '''

    # 计算训练集的最大值，最小值和平均值

    maxmums, minmums, avgs = training_data.max(axis=0), training_data.min(axis=0),\
                             training_data.sum(axis=0) / training_data.shape[0]

    # 归一化

    for i in range(feature_num):
        data[:,i] = (data[:,i] - avgs[i]) / (maxmums[i] - minmums[i])

    # 数据划分
    train_data = data[:offset]
    test_data = data[offset:]
    return train_data,test_data


# 获取数据

# training_data,test_data = load_data()
# x = training_data[:,:-1]
# y = training_data[:,-1:]

# 查看数据
# print(y[0])

# 训练模型的设计 z = wx + b

class Network():
    def __init__(self,num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)  # 设置后np的random模块每次生成的随机数相同
        self.w = np.random.randn(num_of_weights,1)
        self.b = 0.

    def forward(self,x):
        z = np.dot(x,self.w) + self.b
        return z

    def lossfunction(self,z,y):
        lossing = z - y
        num_samples = lossing.shape[0]
        cost = lossing * lossing
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self,x,y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w,axis=0)
        gradient_w = gradient_w[:,np.newaxis]

        gradient_b = (z-y)
        gradient_b = np.mean(gradient_b)

        return gradient_w, gradient_b

    def update(self,gradient_w,gradient_b,eat = 0.01):
        self.w = self.w - eat * gradient_w
        self.b = self.b - eat * gradient_b

    def train(self,x,y,iterations,eta = 0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.lossfunction(z,y)
            gradient_w, gradient_b  = self.gradient(x,y)
            self.update(gradient_w,gradient_b,eta)
            losses.append(L)
            if  (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses




# 获取数据

train_data,test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]

# 初始化网络

net = Network(13)
num_iterations = 1000

# 启动训练

losses = net.train(x,y,iterations=num_iterations,eta = 0.01)

# 画出损失函数

plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x,plot_y)
plt.show()

