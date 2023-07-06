from cgitb import reset
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from resnetcifar import ResNet18_cifar10, ResNet50_cifar10
import sys

#import pytorch_lightning as pl


class MLP_header(nn.Module):
    def __init__(self,):
        super(MLP_header, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        #projection
        # self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class FcNet(nn.Module):
    """
    Fully connected network for MNIST classification
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.dims = [self.input_dim]
        self.dims.extend(hidden_dims)
        self.dims.append(self.output_dim)

        self.layers = nn.ModuleList([])

        for i in range(len(self.dims) - 1):
            ip_dim = self.dims[i]
            op_dim = self.dims[i + 1]
            self.layers.append(
                nn.Linear(ip_dim, op_dim, bias=True)
            )

        self.__init_net_weights__()

    def __init_net_weights__(self):

        for m in self.layers:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Do not apply ReLU on the final layer
            if i < (len(self.layers) - 1):
                x = nn.ReLU(x)

            # if i < (len(self.layers) - 1):  # No dropout on output layer
            #     x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x


class FCBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(FCBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGGConvBlocks(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, num_classes=10):
        super(VGGConvBlocks, self).__init__()
        self.features = features
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FCBlockVGG(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(FCBlockVGG, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        #out = self.conv1(x)
        #out = self.relu(out)
        #out = self.pool(out)
        #out = self.conv2(out)
        #out = self.relu(out)
        #out = self.pool(out)
        #out = out.view(-1, 16 * 5 * 5)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN_header_split(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNN_header, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(84, n_classes)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2()
        x = x.view(-1, 16 * 5 * 5)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

# a simple perceptron model for generated 3D data
class PerceptronModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(PerceptronModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        x = self.fc1(x)
        return x


class SimpleCNNMNIST_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST_header, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return x, 0, y


class SimpleCNNContainer(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNContainer, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims

        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers

        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


############## LeNet for MNIST ###################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.ceriation = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LeNetContainer(nn.Module):
    def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(LeNetContainer, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, 1)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = self.fc1(x)
        x = self.fc2(x)
        return x



### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNN(nn.Module):
    def __init__(self, output_dim=10):
        super(ModerateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNNCeleba(nn.Module):
    def __init__(self):
        super(ModerateCNNCeleba, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 4096)
        x = self.fc_layer(x)
        return x


class ModerateCNNMNIST(nn.Module):
    def __init__(self):
        super(ModerateCNNMNIST, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


class ModerateCNNContainer(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(ModerateCNNContainer, self).__init__()

        ##
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def forward_conv(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x


class ModelFedCon(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelFedCon, self).__init__()

        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            # basemodel = ResNet50_cifar10()
            basemodel = models.resnet50(pretrained=False)
            basemodel.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            basemodel.maxpool = torch.nn.Identity()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            # basemodel = ResNet18_cifar10()
            basemodel=models.resnet18(pretrained=False)
            basemodel.conv1=torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
            basemodel.maxpool=torch.nn.Identity()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18_7":
            basemodel = models.resnet18(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        #summary(self.features.to('cuda:0'), (3,32,32))
        #print("features:", self.features)
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.squeeze()
        #print("h after:", h)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y


class ModelFedCon_noheader(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelFedCon_noheader, self).__init__()
        print('no header')
        if base_model == "resnet50":
            basemodel = models.resnet50(pretrained=False)
            basemodel.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            basemodel.maxpool = torch.nn.Identity()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet50_7":
            basemodel = models.resnet50(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18":
            basemodel = models.resnet18(pretrained=False)
            basemodel.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            basemodel.maxpool = torch.nn.Identity()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18_7":
            basemodel = models.resnet18(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'resnet18_7_gn':
            basemodel = models.resnet18(pretrained=False)
            # Change BN to GN 
            basemodel.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            # assert len(dict(basemodel.named_parameters()).keys()) == len(basemodel.state_dict().keys()), 'More BN layers are there...'
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'resnet18_gn':
            basemodel = models.resnet18(pretrained=False)
            basemodel.conv1=torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
            basemodel.maxpool=torch.nn.Identity()
            # Change BN to GN 
            basemodel.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            # assert len(dict(basemodel.named_parameters()).keys()) == len(basemodel.state_dict().keys()), 'More BN layers are there...'
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'resnet50_gn':
            basemodel = models.resnet50(pretrained=False)
            basemodel.conv1=torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
            basemodel.maxpool=torch.nn.Identity()
            basemodel.bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            
            basemodel.layer1[0].bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn3=nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer1[0].downsample[1]=nn.GroupNorm(num_groups = 2, num_channels = 256)
            
            basemodel.layer1[1].bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn3=nn.GroupNorm(num_groups = 2, num_channels = 256)
            
            basemodel.layer1[2].bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[2].bn2=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[2].bn3=nn.GroupNorm(num_groups = 2, num_channels = 256)
            
            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer2[2].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[2].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[2].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer2[3].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[3].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[3].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            
            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[2].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[2].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[2].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[3].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[3].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[3].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[4].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[4].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[4].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[5].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[5].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[5].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            
            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            basemodel.layer4[2].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[2].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[2].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            assert len(dict(basemodel.named_parameters()).keys()) == len(basemodel.state_dict().keys()), 'More BN layers are there...'

            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features

        #summary(self.features.to('cuda:0'), (3,32,32))
        #print("features:", self.features)
        # projection MLP
        # self.l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.squeeze()
        #print("h after:", h)
        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)

        y = self.l3(h)
        return h, h, y

class ModelTemplate(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelTemplate, self).__init__()
        print('model template')
        self.n_classes = n_classes
        if base_model == "resnet50":
            basemodel = models.resnet50(pretrained=False)
            basemodel.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            basemodel.maxpool = torch.nn.Identity()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet50_7":
            basemodel = models.resnet50(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18":
            basemodel = models.resnet18(pretrained=False)
            basemodel.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            basemodel.maxpool = torch.nn.Identity()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18_7":
            basemodel = models.resnet18(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'resnet18_7_gn':
            basemodel = models.resnet18(pretrained=False)
            # Change BN to GN 
            basemodel.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            # assert len(dict(basemodel.named_parameters()).keys()) == len(basemodel.state_dict().keys()), 'More BN layers are there...'
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'resnet18_gn':
            basemodel = models.resnet18(pretrained=False)
            basemodel.conv1=torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
            basemodel.maxpool=torch.nn.Identity()
            # Change BN to GN 
            basemodel.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            # assert len(dict(basemodel.named_parameters()).keys()) == len(basemodel.state_dict().keys()), 'More BN layers are there...'
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'resnet50_gn':
            basemodel = models.resnet50(pretrained=False)
            basemodel.conv1=torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
            basemodel.maxpool=torch.nn.Identity()
            basemodel.bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            
            basemodel.layer1[0].bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn3=nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer1[0].downsample[1]=nn.GroupNorm(num_groups = 2, num_channels = 256)
            
            basemodel.layer1[1].bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn3=nn.GroupNorm(num_groups = 2, num_channels = 256)
            
            basemodel.layer1[2].bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[2].bn2=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[2].bn3=nn.GroupNorm(num_groups = 2, num_channels = 256)
            
            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer2[2].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[2].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[2].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer2[3].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[3].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[3].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            
            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[2].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[2].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[2].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[3].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[3].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[3].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[4].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[4].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[4].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[5].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[5].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[5].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            
            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            basemodel.layer4[2].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[2].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[2].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            assert len(dict(basemodel.named_parameters()).keys()) == len(basemodel.state_dict().keys()), 'More BN layers are there...'

            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features

        # Template
        self.template = nn.Parameter(nn.Linear(num_ftrs, n_classes).state_dict()['weight'], requires_grad=True)     # shape [C, d]

    # def forward(self, x):
    #     h = self.features(x)
    #     h = h.squeeze()         # B x d

    #     y_dot = torch.mm(h, self.template.T)    # B x C
    #     h_norm = torch.linalg.norm(h, dim=1).unsqueeze(-1)    # B x 1
    #     t_norm = torch.linalg.norm(self.template, dim=1).unsqueeze(-1)    # C x 1
    #     norm_dot = torch.mm(h_norm, t_norm.T)
    #     y = torch.div(y_dot,norm_dot)/0.1
    #     return h, h, y

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()         # B x d

        if len(h.size())<2:
            h = h.unsqueeze(0)

        h_extend = h.unsqueeze(1).repeat(1,self.n_classes,1)
        template = self.template.unsqueeze(0).repeat(h_extend.shape[0],1,1)

        l2_distance = -torch.pow((h_extend-template), 2)
        y = torch.sum(l2_distance, dim=-1)
        return h, h, y

if __name__=='__main__':
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for p in model.parameters():
            print(type(p[0]))
        return total_num, trainable_num
    model = ModelFedCon_noheader('simple-cnn', out_dim=0, n_classes=10, net_configs=None)
    print(get_parameter_number(model))