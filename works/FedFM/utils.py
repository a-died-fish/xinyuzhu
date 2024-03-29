import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
import sys

from model import *
from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom, ImageFolder_WIKI, ImageFolder_CINIC10, ImageFolder_HAM10000, FashionMNIST_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    fmnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    fmnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = fmnist_train_ds.data, fmnist_train_ds.target
    X_test, y_test = fmnist_test_ds.data, fmnist_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'/val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_wiki_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_WIKI(datadir+'/train.npy', transform=transform)
    xray_test_ds = ImageFolder_WIKI(datadir+'/test.npy', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([float(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([float(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_cinic10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_CINIC10(datadir+'/train/', transform=transform)
    xray_test_ds = ImageFolder_CINIC10(datadir+'/sampled_test/', transform=transform)   # 欠采样测试集，大小为10000

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_HAM10000_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_HAM10000(datadir+'/train_new/', transform=transform)
    xray_test_ds = ImageFolder_HAM10000(datadir+'/val_new/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    num_classes = int(y_train.max()) + 1

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_dict[net_i] = tmp
        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate(
                        (net_cls_counts_npy, tmp_npy), axis=0)
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1,num_classes))


    data_list=[]
    for net_id, data in net_cls_counts_dict.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts_dict))

    print(net_cls_counts_npy)
    return net_cls_counts_npy


def partition_regression_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    if dataset == 'wiki':
        X_train, y_train, X_test, y_test = load_wiki_data(datadir)
        age_partition = [1, 24, 29, 37, 52, 100]
    
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    
    elif partition == "noniid-labeldir" or partition == "noniid":
        for i in range(len(age_partition)-1):
            mask = np.logical_and((y_train>=age_partition[i]), (y_train<age_partition[i+1]))
            y_train[mask] = i

        K = len(age_partition)-1
        N = y_train.shape[0]
        net_dataidx_map = {}
        min_size = 0
        min_require_size = 10

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            print(len(net_dataidx_map[j]))
        class_dis = np.zeros((n_parties, K))

        for j in range(n_parties):
            for m in range(K):
                class_dis[j,m] = (np.array(y_train[idx_batch[j]])==m).sum()
        print(class_dis)
    
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4, n_niid_parties=5):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'cinic10' or dataset == 'cinic10_val':
        X_train, y_train, X_test, y_test = load_cinic10_data(datadir)
    elif dataset == 'ham10000':
        X_train, y_train, X_test, y_test = load_HAM10000_data(datadir)

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100
        elif dataset == 'ham10000':
            K = 7

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            print(len(net_dataidx_map[j]))
        class_dis = np.zeros((n_parties, K))

        for j in range(n_parties):
            for m in range(K):
                class_dis[j,m] = (np.array(y_train[idx_batch[j]])==m).sum()
        print(class_dis)

    elif partition=='noniid-2' and dataset=='cifar10':
        num_per_shard = 100
        num_shards = n_train//num_per_shard
        idx_shard = [i for i in range(num_shards)]
        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(num_shards*num_per_shard).astype(int)
        # labels = dataset.train_labels.numpy()
        labels = y_train

        print(n_train)
        print(labels.max()+1)

        # sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)     # each class has 5000 samples
        # print((idxs_labels[1, :]==6).sum())
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]                    # index follows the sequnce of label

        # divide and assign
        if n_parties==10:
            num_shard_per_class = num_shards//(labels.max()+1)
            print('User_class_distribution:')
            for i in range(n_parties):
                small_shards = 3            # 小类占比
                chosen_set = set(range(i*num_shard_per_class,(i+1)*num_shard_per_class-10*small_shards))
                for j in range(10):
                    for q in range(small_shards):
                        chosen_set.add((j+1)*num_shard_per_class-10*small_shards+i*small_shards+q)
                # print(chosen_set)

                class_dict = {}
                for j in range(10):
                    class_dict[j] = 0

                for chosen in chosen_set:
                    net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[chosen*num_per_shard:(chosen+1)*num_per_shard]), axis=0)
                    class_dict[idxs_labels[1,chosen*num_per_shard]] += 1

                np.random.shuffle(net_dataidx_map[i])
                net_dataidx_map[i] = list(net_dataidx_map[i])
                print(class_dict)

    elif partition=='noniid-2' and dataset=='cinic10':
        num_per_shard = 100
        num_shards = n_train//num_per_shard
        idx_shard = [i for i in range(num_shards)]
        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(num_shards*num_per_shard).astype(int)
        # labels = dataset.train_labels.numpy()
        labels = y_train

        print(n_train)
        print(labels.max()+1)

        # sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)     # each class has 5000 samples
        # print((idxs_labels[1, :]==6).sum())
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]                    # index follows the sequnce of label

        # divide and assign
        if n_parties==10:
            num_shard_per_class = num_shards//(labels.max()+1)
            print('User_class_distribution:')
            for i in range(n_parties):
                small_shards = 5            # 小类占比
                chosen_set = set(range(i*num_shard_per_class,(i+1)*num_shard_per_class-10*small_shards))
                for j in range(10):
                    for q in range(small_shards):
                        chosen_set.add((j+1)*num_shard_per_class-10*small_shards+i*small_shards+q)
                # print(chosen_set)

                class_dict = {}
                for j in range(10):
                    class_dict[j] = 0

                for chosen in chosen_set:
                    net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[chosen*num_per_shard:(chosen+1)*num_per_shard]), axis=0)
                    class_dict[idxs_labels[1,chosen*num_per_shard]] += 1

                np.random.shuffle(net_dataidx_map[i])
                net_dataidx_map[i] = list(net_dataidx_map[i])
                print(class_dict)

    elif partition=='noniid-2' and dataset=='cifar100':
        num_per_shard = 10
        num_shards = n_train//num_per_shard
        idx_shard = [i for i in range(num_shards)]
        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(num_shards*num_per_shard).astype(int)
        # labels = dataset.train_labels.numpy()
        labels = y_train

        print(n_train)
        print(labels.max()+1)

        # sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)     # each class has 5000 samples
        # print((idxs_labels[1, :]==6).sum())
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]                    # index follows the sequnce of label

        # divide and assign
        if n_parties==10:
            num_shard_per_class = num_shards//(labels.max()+1)
            print('User_class_distribution:')
            whole_set = set()
            for i in range(n_parties):
                small_shards = 3            # 小类占比

                chosen_class = i*10
                chosen_set = set(range(chosen_class*num_shard_per_class,(chosen_class+1)*num_shard_per_class-10*small_shards))
                for chosen_class in range(i*10+1,(i+1)*10):
                    chosen_set = chosen_set | set(range(chosen_class*num_shard_per_class,(chosen_class+1)*num_shard_per_class-10*small_shards))
                
                for j in range(100):
                    for q in range(small_shards):
                        chosen_set.add((j+1)*num_shard_per_class-10*small_shards+i*small_shards+q)

                class_dict = {}
                for j in range(100):
                    class_dict[j] = 0

                for chosen in chosen_set:
                    net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[chosen*num_per_shard:(chosen+1)*num_per_shard]), axis=0)
                    class_dict[idxs_labels[1,chosen*num_per_shard]] += 1

                np.random.shuffle(net_dataidx_map[i])
                # net_dataidx_map[i].astype(int)
                # print(net_dataidx_map[i])
                # print(net_dataidx_map[i].shape)
                net_dataidx_map[i] = list(net_dataidx_map[i])
                # print(len(net_dataidx_map[i]))
                # print(net_dataidx_map[i])
                print(class_dict)
    
    elif partition=='noniid-2' and dataset=='tinyimagenet':
        num_per_shard = 10
        num_shards = len(y_train)//num_per_shard
        idx_shard = [i for i in range(num_shards)]
        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(num_shards*num_per_shard).astype(int)
        # labels = dataset.train_labels.numpy()
        labels = y_train

        print(len(y_train))
        print(labels.max()+1)

        # sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)     # each class has 5000 samples
        # print((idxs_labels[1, :]==6).sum())
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]                    # index follows the sequnce of label

        # divide and assign
        if n_parties==10:
            num_shard_per_class = num_shards//(labels.max()+1)
            print('User_class_distribution:')
            whole_set = set()
            for i in range(n_parties):
                small_shards = 3            # 小类占比

                chosen_class = i*20
                chosen_set = set(range(chosen_class*num_shard_per_class,(chosen_class+1)*num_shard_per_class-10*small_shards))
                for chosen_class in range(i*20+1,(i+1)*20):
                    chosen_set = chosen_set | set(range(chosen_class*num_shard_per_class,(chosen_class+1)*num_shard_per_class-10*small_shards))
                
                for j in range(200):
                    for q in range(small_shards):
                        chosen_set.add((j+1)*num_shard_per_class-10*small_shards+i*small_shards+q)

                class_dict = {}
                for j in range(200):
                    class_dict[j] = 0

                for chosen in chosen_set:
                    net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[chosen*num_per_shard:(chosen+1)*num_per_shard]), axis=0)
                    class_dict[idxs_labels[1,chosen*num_per_shard]] += 1

                np.random.shuffle(net_dataidx_map[i])
                print(class_dict)
        else:
            raise NotImplementedError()

    elif (partition.startswith('noniid-3')) and dataset=='cifar10':     # noniid-3-2: missing 2
        missing_num=int(partition[-1])
        net_dataidx_map = {}
        idxs = np.arange(len(y_train)).astype(int)
        labels = y_train
        num_per_class = len(y_train) // (labels.max()+1)

        rest_num = (labels.max()+1) - missing_num
        num_per_class_client = num_per_class // rest_num

        # sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)     # each class has 5000 samples
        # print((idxs_labels[1, :]==6).sum())
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]                    # index follows the sequnce of label

        class_set_dict = {}
        for i in range((labels.max()+1)):
            class_set_dict[i] = idxs[i*num_per_class:(i+1)*num_per_class]

        if n_parties==10:
            for i in range(n_parties):
                for j in range(rest_num):
                    choosing_class = (i+j)%10
                    choosing_set = set(np.random.choice(class_set_dict[choosing_class], num_per_class_client,
                                                        replace=False))
                    if i in net_dataidx_map:
                        net_dataidx_map[i] = net_dataidx_map[i] | choosing_set
                    else:
                        net_dataidx_map[i] = choosing_set
                    class_set_dict[choosing_class] = list(set(class_set_dict[choosing_class]) - choosing_set)
                net_dataidx_map[i] = list(net_dataidx_map[i])
                np.random.shuffle(net_dataidx_map[i])
    
    elif partition=='noniid-4' and dataset=='cifar10':

        labels = y_train
        
        num_non_iid_client = n_niid_parties                              # has 2 classes, should satisfy num_non_iid_client%5=0
        num_iid_client =  n_parties-num_non_iid_client      # has 10 classes
        num_classes = int(labels.max()+1)
        num_sample_per_client = n_train//n_parties
        num_sample_per_class = n_train//num_classes
        num_per_shard = int(n_train/num_classes/(num_non_iid_client+num_iid_client))     # num_non_iid_client+num_iid_client: non_iid_client has 2 classes，while iid_client has 10 classes, so non_iid_client has 5 times data per class

        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(n_train).astype(int)

        # sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)     # each class has 5000 samples
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]                    # index follows the sequnce of label

        for i in range(num_non_iid_client):
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i)%10)*num_sample_per_class+num_per_shard*(i//5)*5:((2*i)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5]), axis=0)
            print(((2*i)%10)*num_sample_per_class+num_per_shard*(i//5)*5,((2*i)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5)
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5)*5:((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5]), axis=0)
            print(((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5)*5,((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5)
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
        
        for i in range(num_non_iid_client,n_parties):
            for j in range(num_classes):
                net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client): \
                                                    j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1)]), axis=0)
                print(j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client),j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1))
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
    
    elif partition=='noniid-5' and dataset=='cifar10':

        labels = y_train
        
        num_non_iid_client = n_niid_parties                 # has 2 classes, should satisfy num_non_iid_client%5=0
        num_iid_client =  n_parties-num_non_iid_client      # has 10 classes
        num_classes = int(labels.max()+1)
        num_sample_per_client = 1000
        num_sample_per_class = n_train//num_classes
        
        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(n_train).astype(int)

        # sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)     # each class has 5000 samples
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]                    # index follows the sequnce of label

        num_per_shard_niid = num_sample_per_client//2
        for i in range(num_non_iid_client):
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i)%10)*num_sample_per_class+num_per_shard_niid*(i//5):((2*i)%10)*num_sample_per_class+num_per_shard_niid*(i//5+1)]), axis=0)
            print(((2*i)%10)*num_sample_per_class+num_per_shard_niid*(i//5),((2*i)%10)*num_sample_per_class+num_per_shard_niid*(i//5+1))
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i+1)%10)*num_sample_per_class+num_per_shard_niid*(i//5):((2*i+1)%10)*num_sample_per_class+num_per_shard_niid*(i//5+1)]), axis=0)
            print(((2*i+1)%10)*num_sample_per_class+num_per_shard_niid*(i//5),((2*i+1)%10)*num_sample_per_class+num_per_shard_niid*(i//5+1))
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
        
        num_per_shard_iid = num_sample_per_client//10
        for i in range(num_non_iid_client,n_parties):
            for j in range(num_classes):
                net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[j*num_sample_per_class+num_per_shard_niid*(num_non_iid_client//5)+num_per_shard_iid*(i-num_non_iid_client): \
                                                    j*num_sample_per_class+num_per_shard_niid*(num_non_iid_client//5)+num_per_shard_iid*(i-num_non_iid_client+1)]), axis=0)
                print(j*num_sample_per_class+num_per_shard_niid*(num_non_iid_client//5)+num_per_shard_iid*(i-num_non_iid_client),j*num_sample_per_class+num_per_shard_niid*(num_non_iid_client//5)+num_per_shard_iid*(i-num_non_iid_client+1))
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
    
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def partition_test_data(traindata_cls_counts, y_test, n_parties):

    n_test = y_test.shape[0]
    
    net_dataidx_map_test = {i: np.array([]).astype(int) for i in range(n_parties)}
    idxs = np.arange(n_test).astype(int)
    labels = y_test

    print(n_test)
    num_classes = int(labels.max()+1)
    print(f'Number of classes : {num_classes}')

    # sort labels
    idxs_labels = np.vstack((idxs, labels)).astype(int)     # each class has 1000 samples
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]                    # index follows the sequnce of label

    train_class_count = np.sum(traindata_cls_counts, axis=0)
    print(train_class_count)
    test_class_count = np.zeros(num_classes)
    for i in range(num_classes):
        test_class_count[i] = np.count_nonzero(labels == i)
    print(test_class_count)
    
    partition_start = 0
    for class_id in range(num_classes):
        for client in range(n_parties):
            if int(traindata_cls_counts[client][class_id]/train_class_count[class_id]*test_class_count[class_id]) >=1:
                net_dataidx_map_test[client] = np.concatenate((net_dataidx_map_test[client], idxs[partition_start:partition_start+int(traindata_cls_counts[client][class_id]/train_class_count[class_id]*test_class_count[class_id])]), axis=0)
                partition_start += int(traindata_cls_counts[client][class_id]/train_class_count[class_id]*test_class_count[class_id])
            else:
                pass
            print(f'class {class_id} {client} {partition_start}')
    
    for client in range(n_parties):
        np.random.shuffle(net_dataidx_map_test[client])
        net_dataidx_map_test[client] = list(net_dataidx_map_test[client])
    
    return net_dataidx_map_test



def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel

def save_features(model, dataloaders, save_dir, round):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    for idx, dataloader in enumerate(dataloaders):
        images, labels_idx = next(iter(dataloader))
        images = images.cuda()
        # labels = labels.cuda()
        with torch.no_grad():
            representation_idx, _, _ = model(images)
        representation_idx = representation_idx.cpu().numpy()
        labels_idx = labels_idx.cpu().numpy()
        if idx==0:
            representations=representation_idx
            labels=labels_idx
        else:
            representations = np.concatenate(
                (representations, representation_idx), axis=0)
            labels = np.concatenate(
                (labels, labels_idx), axis=0)
    np.save(os.path.join(save_dir, 'global_model_rep'+str(round)+'.npy'),representations)
    np.save(os.path.join(save_dir, 'global_model_label'+str(round)+'.npy'),labels)

    if was_training:
        model.train()


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    _, _, out = model(x)
                    # print(out.shape, target.shape)
                    if len(target)==1:
                        out= out.unsqueeze(0)
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                _,_,out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss

def compute_loss_regression(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    if device == 'cpu':
        criterion = nn.MSELoss()
    elif "cuda" in device.type:
        criterion = nn.MSELoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    if device != 'cpu':
                        x, target = x.cuda(), target.cuda()
                    _, _, out = model(x)
                    target = (target/100).to(torch.float32)
                    if len(target)==1:
                        out= out.unsqueeze(0)
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    x, target = x.cuda(), target.cuda()
                _,_,out = model(x)
                target = (target/100).to(torch.float32)
                loss = criterion(out, target)
                loss_collector.append(loss.item())
            avg_loss = sum(loss_collector) / len(loss_collector)
    if was_training:
        model.train()
    return avg_loss



def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda()
    return model


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, drop_last=True, noise_level=0):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            # transform_train = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Lambda(lambda x: F.pad(
            #         Variable(x.unsqueeze(0), requires_grad=False),
            #         (4, 4, 4, 4), mode='reflect').data.squeeze()),
            #     transforms.ToPILImage(),
            #     transforms.ColorJitter(brightness=noise_level),
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            # data prep for test set
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     normalize])
            transform_test=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

        return train_dl, test_dl, train_ds, test_ds

    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir+'/train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'/val/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

        return train_dl, test_dl, train_ds, test_ds
    
    elif dataset == 'wiki':
        dl_obj = ImageFolder_WIKI
        transform_train = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_ds = dl_obj(datadir+'/train.npy', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'/test.npy', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

        return train_dl, test_dl, train_ds, test_ds
    
    elif dataset == 'cinic10_val':
        dl_obj = ImageFolder_CINIC10

            # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train=transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])

        transform_test=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])
        
        train_ds = dl_obj(datadir+'/train/', dataidxs=dataidxs, transform=transform_train)
        val_ds = dl_obj(datadir+'/valid/', transform=transform_test)
        test_ds = dl_obj(datadir+'/test/', transform=transform_test)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True)
        val_dl = data.DataLoader(dataset=val_ds, batch_size=test_bs, shuffle=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

        return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds
    
    elif dataset == 'cinic10':
        dl_obj = ImageFolder_CINIC10

            # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train=transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])

        transform_test=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])
        
        train_ds = dl_obj(datadir+'/train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'/sampled_test/', transform=transform_test)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

        return train_dl, test_dl, train_ds, test_ds
    
    elif dataset == 'ham10000':
        dl_obj = ImageFolder_HAM10000

            # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train=transforms.Compose([
            transforms.Resize((56, 56)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        transform_test=transforms.Compose([
            transforms.Resize((56, 56)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_ds = dl_obj(datadir+'/train_new/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'/val_new/', transform=transform_test)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

        return train_dl, test_dl, train_ds, test_ds
    
    elif dataset == 'fmnist':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081)),
        ])
        train_ds = FashionMNIST_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = FashionMNIST_truncated(datadir, train=False, transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

        return train_dl, test_dl, train_ds, test_ds
