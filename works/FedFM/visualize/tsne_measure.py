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

from sklearn import metrics
from sklearn.cluster import KMeans

#0224130926_us10_ep5_bs64
#0226013534_us10_ep10_bs64 fedfm
#0226213516_us10_ep10_bs64 fedavg
#0226213546_us10_ep10_bs64 fedprox
#0226213552_us10_ep10_bs64 fednova


# method_list = ['fedavg', 'fedprox', 'fednova', 'moon', 'fedfm']
method_features_file_dict = {
    'fedavg': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226213516_us10_ep10_bs64/global_model_rep89.npy',
    'fedavgm': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0203-40/global_model_rep89.npy',
    'fedprox_89': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226213546_us10_ep10_bs64/global_model_rep89.npy',
    'fedprox_59': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226213546_us10_ep10_bs64/global_model_rep59.npy',
    'scaffold': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-41/global_model_rep89.npy',
    'feddyn_79': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-54/global_model_rep79.npy',
    'feddyn_89': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-54/global_model_rep89.npy',
    'feddyn_99': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-54/global_model_rep99.npy',
    'fednova': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226213552_us10_ep10_bs64/global_model_rep89.npy',
    'moon': '/GPFS/data/zhenyangni/moonfm/feature/feature1_90.npy',
    'fedfm_79': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226013534_us10_ep10_bs64/global_model_rep79.npy',
    'fedfm_89': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226013534_us10_ep10_bs64/global_model_rep89.npy',
    'fedfm_99': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226013534_us10_ep10_bs64/global_model_rep99.npy',
}

method_labels_file_dict = {
    'fedavg': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226213516_us10_ep10_bs64/global_model_label89.npy',
    'fedavgm': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0203-40/global_model_label89.npy',
    'fedprox_89': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226213546_us10_ep10_bs64/global_model_label89.npy',
    'fedprox_59': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226213546_us10_ep10_bs64/global_model_label89.npy',
    'scaffold': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-41/global_model_label89.npy',
    'feddyn_79': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-54/global_model_label79.npy',
    'feddyn_89': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-54/global_model_label89.npy',
    'feddyn_99': '/GPFS/data/ruiye/fssl/MOON_1/logs/cifar10/features/2022-08-31-0204-54/global_model_label99.npy',
    'fednova': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226213552_us10_ep10_bs64/global_model_label89.npy',
    'moon': '/GPFS/data/zhenyangni/moonfm/feature/label_90.npy',
    'fedfm_79': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226013534_us10_ep10_bs64/global_model_label79.npy',
    'fedfm_89': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226013534_us10_ep10_bs64/global_model_label89.npy',
    'fedfm_99': '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226013534_us10_ep10_bs64/global_model_label99.npy',
}

'''
# Rreparation: save tsne file
for method in method_features_file_dict.keys():
    point = np.load(method_features_file_dict[method])
    label = np.load(method_labels_file_dict[method])
    print(f'Processing : {method} | shape : {point.shape}')
    if label.shape[0]==4000:
        mask = np.array(range(label.shape[0]))
        mask = mask%400<200
        point = point[mask]
        label = label[mask]
        print(f'Filtering {method}')
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(point)
    np.save('tsne_file/'+method+'_point.npy', X_2d)
'''


nmi_list = []
silhouette_list_1 = []
silhouette_list_2 = []
for method in method_features_file_dict.keys():
    point = np.load(f'tsne_file/{method}_point.npy')
    label = np.load(method_labels_file_dict[method])
    if label.shape[0]==4000:
        mask = np.array(range(label.shape[0]))
        mask = mask%400<200
        label = label[mask]
        print(f'Filtering {method}')
    silhouette_list_1.append(metrics.silhouette_score(point, label, metric='euclidean'))
    kmeans_model = KMeans(n_clusters=10, random_state=1).fit(point)
    labels_pred = kmeans_model.labels_
    silhouette_list_2.append(metrics.silhouette_score(point, labels_pred, metric='euclidean'))
    nmi_list.append(metrics.normalized_mutual_info_score(label, labels_pred))
    print(f'Processing : {method} | shape : {point.shape} | ss1 : {silhouette_list_1[-1]} | ss2 : {silhouette_list_2[-1]} | nmi : {nmi_list[-1]}' )

    plt.figure(figsize=(5, 5))
    color_bank = ['b', 'c', 'g', 'k', 'm', 'r', 'y', 'gray', 'darkred', 'deeppink']
    for i in range(point.shape[0]):
        plt.scatter(point[i,0], point[i, 1], c=color_bank[label[i]], s=10)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f'tsne_file/tsne_{method}.png', dpi=300)
    plt.close()

print(nmi_list)
print(silhouette_list_1)
print(silhouette_list_2)


'''
method_list = ['fedavg', 'fedprox', 'fednova', 'fedfm']
exp_name_list = ['0226213516_us10_ep10_bs64', '0226213546_us10_ep10_bs64', '0226213552_us10_ep10_bs64', '0226013534_us10_ep10_bs64']
epoch_list = [89, 89, 89, 89]

for i in range(len(epoch_list)):
    exp_time = exp_name_list[i]
    epoch = epoch_list[i]
    dir_path = '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/'+exp_time
    rep_path = os.path.join(dir_path, 'global_model_rep'+str(epoch)+'.npy')
    label_path = os.path.join(dir_path, 'global_model_label'+str(epoch)+'.npy')
    rep = np.load(rep_path)
    label = np.load(label_path)
    mask = np.array(range(rep.shape[0]))
    mask = mask%400<200
    rep = rep[mask]
    label = label[mask]
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(rep)
    np.save('tsne_file/'+method_list[i]+'_point.npy', X_2d)
    np.save('tsne_file/'+method_list[i]+'_label.npy', label)
    print(metrics.silhouette_score(X_2d, label, metric='euclidean'))

epoch = 90
dir_path = '/GPFS/data/zhenyangni/moonfm/feature'
rep_path = os.path.join(dir_path, 'feature1_'+str(epoch)+'.npy')
label_path = os.path.join(dir_path, 'label_'+str(epoch)+'.npy')
rep = np.load(rep_path)
label = np.load(label_path)
mask = np.array(range(rep.shape[0]))
mask = mask%400<200
rep = rep[mask]
label = label[mask]
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(rep)
np.save('tsne_file/moon_point.npy', X_2d)
np.save('tsne_file/moon_label.npy', label)
print(metrics.silhouette_score(X_2d, label, metric='euclidean'))
'''

'''
epoch = 89
dir_path = '/GPFS/data/ruiye/fssl/Federated-Learning-PyTorch/save_byep/cifar10/0226013534_us10_ep10_bs64'
rep_path = os.path.join(dir_path, 'global_model_rep'+str(epoch)+'.npy')
label_path = os.path.join(dir_path, 'global_model_label'+str(epoch)+'.npy')
tsne_save_path = os.path.join(dir_path+'/figs', str(epoch)+'_tsne_2.png')

representations = np.load(rep_path)
labels = np.load(label_path)

print(representations.shape)

len_rep = representations.shape[0]

rep = representations
label = labels

mask = np.array(range(rep.shape[0]))
mask = mask%400<200
rep = rep[mask]
label = label[mask]

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(rep)

plt.figure(figsize=(7, 7))

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
# moon tsne
epoch = 80
dir_path = '/GPFS/data/zhenyangni/moonfm/feature'
rep_path = os.path.join(dir_path, 'feature1_'+str(epoch)+'.npy')
label_path = os.path.join(dir_path, 'label_'+str(epoch)+'.npy')
tsne_save_path = 'vis_file/moon_'+str(epoch)+'_1.png'

representations = np.load(rep_path)
labels = np.load(label_path)

print(representations.shape)

len_rep = representations.shape[0]

rep = representations
label = labels

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(rep)

plt.figure(figsize=(5, 5))

color_bank = ['b', 'c', 'g', 'k', 'm', 'r', 'y', 'gray', 'darkred', 'deeppink']

for i in range(rep.shape[0]):
    plt.scatter(X_2d[i,0], X_2d[i, 1], c=color_bank[label[i]], s=10)

plt.xticks([])
plt.yticks([])
plt.savefig(tsne_save_path)
plt.close()
'''