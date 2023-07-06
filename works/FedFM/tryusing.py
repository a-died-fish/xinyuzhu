import torchvision
import torch
from torchsummary import summary
import numpy as np
# import math

# model = torchvision.models.resnet18()
# # model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# # model.maxpool = torch.nn.Identity()
# model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
# model = model.cuda()

# # # model = ResNet18_cifar10().cuda()
# # print(model)
# summary(model, (3, 32, 32))

template = torch.nn.Parameter(torch.nn.Linear(512, 10).state_dict()['weight'], requires_grad=True)
a = torch.rand(64, 512)
print(torch.linalg.norm(a, dim=1).size())


# perfedfm_gen = np.array([58.41, 58.26, 58.52])
# perfedfm_per = np.array([77.32, 76.91, 77.21])

# print(np.mean(perfedfm_gen))
# print(np.mean(perfedfm_per))
# print(np.std(perfedfm_gen))
# print(np.std(perfedfm_per))

# fedproto_gen = np.array([35.91,35.67,35.75,35.99])
# fedproto_per = np.array([38.70,38.49,38.37,38.83])

# print(np.mean(fedproto_gen))
# print(np.mean(fedproto_per))
# print(np.std(fedproto_gen))
# print(np.std(fedproto_per))

# print(np.random.normal(loc=0,scale=0.1,size=(1,10)))


'''
results_dict = {
    'cifar10_fedavg_1': np.array([67.14, 65.72, 67.22]),
    'cifar10_fedprox_1': np.array([67.32, 67.17, 67.09, 66.79, 66.61]),
    'cifar10_scaffold_1': np.array([69.16, 70.39, 70.17]),
    'cifar10_fednova_1': np.array([67.37, 67.37, 65.66]),
    'cifar10_moon_1': np.array([68.00, 67.63, 67.92, 67.19, 67.96]),
    'cifar10_fedproto_1': np.array([67.98, 67.24, 66.78]),
    'cifar10_fedavgm_1': np.array([67.88, 66.39, 66.66, 67.28]),
    'cifar10_feddyn_1':np.array([68.07, 68.13, 68.98, 68.25, 68.15]),
    'cifar10_fedfm_1': np.array([73.01, 72.12, 72.67, 73.21, 72.94]),
    'cifar10_fedavg_2': np.array([68.80, 69.84, 69.78]),
    'cifar10_fedprox_2': np.array([69.44, 69.29, 70.11, 69.25, 68.99]),
    'cifar10_scaffold_2': np.array([71.41, 71.79, 71.23]),
    'cifar10_fednova_2': np.array([68.80, 69.84, 69.78]),
    'cifar10_moon_2': np.array([71.24, 71.11, 71.13, 71.31, 70.67]),
    'cifar10_fedavgm_2': np.array([69.46, 68.86, 69.04, 68.92]),
    'cifar10_fedfm_2': np.array([74.68, 74.23, 74.66]),
    'cifar10_fedfm_l2_2': np.array([70.57, 70.36, 70.71]),
    'cifar10_feddyn_2': np.array([67.47, 67.5, 67.67, 67.87]),
    'cifar10_fedavg_3': np.array([70.70, 70.16, 70.64]),
    'cifar10_fedprox_3_2': np.array([70.51, 71.21, 70.44]),
    'cifar10_fedprox_3_3': np.array([69.52, 69.87, 70.07]),
    'cifar10_fedprox_3_5': np.array([67.90, 66.58, 67.28]),
    'cifar10_fedprox_3_8': np.array([42.59, 44.25, 43.42]),
    'cifar10_scaffold_3_2': np.array([73.28, 72.56, 72.98]),
    'cifar10_scaffold_3_3': np.array([72.30, 72.66, 72.84]),
    'cifar10_scaffold_3_5': np.array([71.39, 71.51, 71.40]),
    'cifar10_scaffold_3_8': np.array([43.29, 43.18, 43.45]),
    'cifar10_moon_3_2': np.array([72.49, 71.95, 72.2]),
    'cifar10_moon_3_3': np.array([71.76, 71.7, 71.25]),
    'cifar10_moon_3_5': np.array([69.24, 68.64, 68.7]),
    'cifar10_moon_3_8': np.array([36.75, 40.97, 40.32]),
    'cifar10_fedavgm_3': np.array([70.26, 69.99, 70.35, 69.56]),
    'cifar10_feddyn_3_2':np.array([67.88, 68.48, 66.93]),
    'cifar10_feddyn_3_3':np.array([68.17, 67.67, 67.51]),
    'cifar10_feddyn_3_5':np.array([68.88, 70.30, 69.40]),
    'cifar10_feddyn_3_7':np.array([65.17]),
    'cifar10_feddyn_3_8':np.array([51.81, 53.34, 53.38]),
    'cifar10_fedfm_3_2': np.array([75.65, 76.16, 75.57]),
    'cifar10_fedfm_3_7': np.array([65.62]),
    'cifar10_fedfm_l2_3': np.array([70.70, 70.99, 71.66, 70.66]),
    'cifar100_fedprox_1': np.array([61.92, 61.94, 61.92, 62.04]),
    'cifar100_moon_1': np.array([62.59, 62.53, 62.26, 62.87]),
    'cifar100_dyn_1': np.array([43.95, 43.62, 42.67]),
    'cifar100_prox_2': np.array([62.54, 61.8, 62.46, 61.98]),
    'cifar100_moon_2': np.array([62.78, 63.11, 63.09, 62.98]),
    'cifar100_dyn_2': np.array([47.37, 46.66, 45.28]),
}

for key in results_dict.keys():
    print(f'{key} : Mean {np.mean(results_dict[key])} | {np.std(results_dict[key])})')
'''

# a = torch.tensor([1,2,3])
# # b = torch.tensor([1,2,3])
# print(a*a)
# print(a**2)
