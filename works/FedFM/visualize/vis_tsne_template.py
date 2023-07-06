#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import sys

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import torch

#0224130926_us10_ep5_bs64
#0226013534_us10_ep10_bs64 fedfm
#0226213516_us10_ep10_bs64 fedavg
#0226213546_us10_ep10_bs64 fedprox
#0226213552_us10_ep10_bs64 fednova
# 2022-08-29-1830-02 fedavgm
# 2022-08-29-1833-58 scaffold
# 2022-08-29-1834-21 feddyn


method = 'fedtemplate-1'
epoch = 99
method_file_dict = {
    'fedavgm':'/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0203-40',
    'scaffold':'/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-41',
    'feddyn':'/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-54',
    'fedfm':'/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226013534_us10_ep10_bs64',
    'fedtemplate-0.001': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-11-22-2127-09',
    'fedtemplate-0.01': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-11-22-2127-02',
    'fedtemplate-1': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-11-22-2126-34',
}
exp_time = (method_file_dict[method].split('/')[-1])
template = np.array(torch.load(f'/GPFS/data/ruiye/fssl/MOON_1/models/fedtemplate/globalmodelexperiment_log-{exp_time}.pth')['template'])

dir_path = method_file_dict[method]
rep_path = os.path.join(dir_path, 'global_model_rep'+str(epoch)+'.npy')
label_path = os.path.join(dir_path, 'global_model_label'+str(epoch)+'.npy')
tsne_save_path = os.path.join('vis_file', f'{method}_{epoch}_tsne.png')

representations = np.load(rep_path)
labels = np.load(label_path)

if representations.shape[0]==4000:
    mask = np.array(range(representations.shape[0]))
    mask = mask%400<200
    representations = representations[mask]
    labels = labels[mask]
    print(f'Filtering {method}')

# calculate mean of feature
total_class = np.max(labels)+1
for label in range(total_class):
    mask = labels==label
    representations_mean = np.mean(representations[mask], axis=0)[np.newaxis,:]
    representations = np.concatenate((representations, representations_mean), axis=0)
    labels = np.concatenate((labels, np.array(label)[np.newaxis]), axis=0)


# template
representations = np.concatenate((representations, template), axis=0)
labels = np.concatenate((labels, np.array(range(10))), axis=0)

print(representations.shape)
print(labels.shape)


total = 0
learn_correct, mean_correct = 0, 0
for i in range(representations.shape[0]-20):
    total+=1
    feature = np.repeat(representations[i][np.newaxis,:], total_class, axis=0)
    l2_distance = -np.power((feature-template), 2)
    y = np.sum(l2_distance, axis=-1)
    if np.argmax(y)==labels[i]:
        learn_correct+=1
    l2_distance = -np.power((feature-representations[representations.shape[0]-20:representations.shape[0]-10]), 2)
    y = np.sum(l2_distance, axis=-1)
    if np.argmax(y)==labels[i]:
        mean_correct+=1

print(f'learn_acc:{learn_correct/total} | mean_acc:{mean_correct/total}')
sys.exit()

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(representations)

plt.figure(figsize=(5, 5))

color_bank = ['b', 'c', 'g', 'k', 'm', 'r', 'y', 'gray', 'darkred', 'deeppink']

for i in range(representations.shape[0]-20):
    plt.scatter(X_2d[i,0], X_2d[i, 1], c=color_bank[labels[i]], s=10)

# plot mean
for i in range(10):
    plt.scatter(X_2d[representations.shape[0]-20+i,0], X_2d[representations.shape[0]-20+i, 1], c=color_bank[i], s=200, marker='*', edgecolor='white')

for i in range(10):
    plt.scatter(X_2d[representations.shape[0]-10+i,0], X_2d[representations.shape[0]-10+i, 1], c=color_bank[i], s=100)

plt.xticks([])
plt.yticks([])

plt.savefig(tsne_save_path, dpi=300)
plt.close()


'''
# moon tsne
epoch = 80
dir_path = '/GPFS/data/zhenyangni/moonfm/feature'
rep_path = os.path.join(dir_path, 'feature1_'+str(epoch)+'.npy')
label_path = os.path.join(dir_path, 'label_'+str(epoch)+'.npy')
tsne_save_path = 'vis_file/moon_'+str(epoch)+'_0829.png'

representations = np.load(rep_path)
labels = np.load(label_path)

print(representations.shape)

len_rep = representations.shape[0]

rep = representations
label = labels

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(rep)

plt.figure(figsize=(5, 5))

# color_bank = ['slateblue', 'navajowhite', 'palegreen', 'violet', 'lavender', 'lightskyblue', 'turquoise', 'silver', 'burlywood', 'lightsalmon']
# color_bank = ['slateblue', 'navajowhite', 'palegreen', 'violet', 'lavender', 'lightskyblue', 'turquoise', 'silver', 'orange', 'lightsalmon']
color_bank = ['b', 'c', 'g', 'k', 'm', 'r', 'y', 'gray', 'darkred', 'deeppink']

for i in range(rep.shape[0]):
    plt.scatter(X_2d[i,0], X_2d[i, 1], c=color_bank[label[i]], s=10)
    # plt.annotate(label[i], xy = (X_2d[i,0], X_2d[i,1]), xytext = (X_2d[i,0], X_2d[i,1]))

plt.xticks([])
plt.yticks([])

# plt.legend()
# plt.title(tsne_save_path.split('.')[0])
# plt.savefig("figs/t-sne_mnist_FSSL_1.png")
plt.savefig(tsne_save_path)
plt.close()
'''


'''
tsne_save_path = os.path.join(fig_dir, 'client_feature_1.png')
plot_tsne(representations[:len_rep//2], labels[:len_rep//2], tsne_save_path)
tsne_save_path = os.path.join(fig_dir, 'client_feature_2.png')
plot_tsne(representations[len_rep//2:], labels[len_rep//2:], tsne_save_path)
'''


'''
global_weights = average_weights(local_weights)
global_model.load_state_dict(global_weights)


for idx in range(solo_number):
    local_model = LocalUpdate(args=args, dataset=train_dataset,
                            idxs=user_groups[idx], logger=logger, dominant_class=user_dominant[idx], num_classes=args.num_classes)
    for j in range(1):
        representation_idx, label_idx = local_model.return_representation(model=global_model)
        if idx==0 and j==0:
            representations=representation_idx
            labels=label_idx
        else:
            representations = np.concatenate(
                (representations, representation_idx), axis=0)
            labels = np.concatenate(
                (labels, label_idx), axis=0)

print(representations.shape)
tsne_save_path = os.path.join(fig_dir, 'global_feature.png')

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(representations)

plt.figure(figsize=(10, 10))

# color_bank = ['b','r']

len_rep = representations.shape[0]

for i in range(len_rep):
    if i<len_rep//2:
        plt.scatter(X_2d[i,0], X_2d[i, 1], c='b')
    else:
        plt.scatter(X_2d[i,0], X_2d[i, 1], c='r')
    plt.annotate(labels[i], xy = (X_2d[i,0], X_2d[i,1]), xytext = (X_2d[i,0], X_2d[i,1]))

plt.legend()
plt.title(tsne_save_path.split('.')[0])
plt.savefig(tsne_save_path)
plt.close()

'''



'''
# representation on server's testing set

testloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.numpy()
        for idx in range(solo_number):
            global_model.load_state_dict(local_weights[idx])
            _, representations = global_model(images, last2_layer=True)
            representations = representations.cpu().numpy()
            tsne_save_path = os.path.join(fig_dir, 'test_client_feature_'+str(idx+1)+'.png')
            plot_tsne(representations, labels, tsne_save_path)
        
        global_model.load_state_dict(global_weights)
        _, representations = global_model(images, last2_layer=True)
        representations = representations.cpu().numpy()
        tsne_save_path = os.path.join(fig_dir, 'test_global_feature.png')
        plot_tsne(representations, labels, tsne_save_path)
        break
'''