from importlib_metadata import distribution
import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import math


from model import *
from utils import *
from torch_utils import *
from opt_utils import *
from feddyn import *

from feature_matching.utils import get_client_centroids_info, get_global_centroids
from feature_matching.loss_f import matching_cross_entropy, matching_l2

from fedtemplate import l2_loss

from distribution_aware.utils import get_distribution_difference


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:2', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')

    parser.add_argument('--print_local_test_acc', type=int, default=1)
    parser.add_argument('--save_feature', type=int, default=0)

    parser.add_argument('--lam_fm', type=float, default=50.0)
    parser.add_argument('--start_ep_fm', type=int, default=20, help='which round to start fm')
    parser.add_argument('--l2match', action='store_true')
    parser.add_argument('--fm_avg_anchor', type=int, default=0, help='equally average')
    parser.add_argument('--cg_tau', type=float, default=0.1, help='tempreture')

    parser.add_argument('--regression', action='store_true')

    # for fedopt
    parser.add_argument('--opt_beta_1', type=float, default=0.0)
    parser.add_argument('--opt_beta_2', type=float, default=0.0)
    parser.add_argument('--opt_global_lr', type=float, default=0.1)
    parser.add_argument('--opt_tau', type=float, default=1e-3)


    # Experiments on FedDisco
    parser.add_argument('--n_niid_parties', type=int, default=5, help='number of niid workers')
    parser.add_argument('--distribution_aware', type=str, default='not', help='Types of distribution aware e.g. division')
    parser.add_argument('--measure_difference', type=str, default='only_iid', help='How to measure difference. e.g. only_iid, cosine')

    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'cinic10', 'cinic10_val'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'ham10000':
        n_classes = 7
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.regression:
        n_classes = 1
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                if args.alg.startswith('fedtemplate'):
                    net = ModelTemplate(args.model, args.out_dim, n_classes, net_configs)
                else:
                    net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    # net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    if args.regression:
        criterion = nn.MSELoss.cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    if args.print_local_test_acc:
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        test_acc = 0.0
    net.to('cpu')

    logger.info(' ** Training complete **')
    return test_acc

def train_net_fm(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, global_centroids, start_fm, device="cpu"):
    # net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            representation,_,out = net(x)

            if start_fm:
                if args.l2match:
                    loss_fm = args.lam_fm * matching_l2(features=representation, labels=target, centroids=global_centroids)
                else:

                    loss_fm = args.lam_fm * matching_cross_entropy(representation, labels=target,
                                                centroids=global_centroids, tao=args.cg_tau)
            else:
                loss_fm = 0.0
            loss_1 = criterion(out, target)
            loss = loss_1+loss_fm
                
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    
    if args.print_local_test_acc:
        # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        # logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        test_acc = 0.0

    net.to('cpu')

    logger.info(' ** Training complete **')
    return test_acc

def train_net_fedproc(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, global_centroids, current_round, device="cpu"):
    # net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()


            representation,_,out = net(x)

            alpha = 1 - current_round/args.comm_round

            loss_fm = alpha * matching_cross_entropy(representation, labels=target,
                                        centroids=global_centroids, tao=args.cg_tau)
            
            loss_ce = (1-alpha) * criterion(out, target)
            loss = loss_fm + loss_ce
                
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    
    if args.print_local_test_acc:
        # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        # logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        test_acc = 0.0

    net.to('cpu')

    logger.info(' ** Training complete **')
    return test_acc

def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    # net = nn.DataParallel(net)
    net.cuda()
    # else:
    #     net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    global_weight_collector = list(global_net.cuda().parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    if args.print_local_test_acc:
        # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        # logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        test_acc = 0.0

    net.to('cpu')
    logger.info(' ** Training complete **')
    # return train_acc, test_acc
    return test_acc

def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):
    # net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda()
    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    if args.print_local_test_acc:
        # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        # logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        test_acc = 0.0

    net.to('cpu')
    logger.info(' ** Training complete **')
    return test_acc

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    # if type(train_dataloader) == type([1]):
    #     pass
    # else:
    #     train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):      # ???
            # x, target = x.to(device), target.to(device)
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            net_para = net.state_dict()
            for key in net_para:
                net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
            net.load_state_dict(net_para)

            cnt += 1
            epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    # 更新c_local
    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    if args.print_local_test_acc:
        # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        # logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        test_acc = 0.0

    net.to('cpu')
    logger.info(' ** Training complete **')
    return test_acc, c_delta_para

def train_net_template_add(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    # net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    if args.regression:
        criterion = nn.MSELoss.cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,h,out = net(x)
            loss_1 = criterion(out, target)
            loss_2 = l2_loss(h, target, net.template)
            loss = loss_1 + args.mu * loss_2
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    if args.print_local_test_acc:
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info('>> Test accuracy: %f' % test_acc)
    else:
        test_acc = 0.0
    net.to('cpu')

    logger.info(' ** Training complete **')
    return test_acc

def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu", current_round=0):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        for name, param in net.named_parameters():
            if not param.requires_grad:
                print(name)
        # print(net.device)

        # print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local=train_dl[net_id]
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        if args.alg in ['fedavg', 'fednova', 'fedopt_adagrad', 'fedopt_adam', 'fedopt_yogi', 'v2_fedopt_adagrad', 'v2_fedopt_adam', 'v2_fedopt_yogi', 'fedtemplate']:
            testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        elif args.alg =='fedfm':
            testacc = train_net_fm(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, global_centroids, current_round>=args.start_ep_fm,
                                        device=device)
        elif args.alg == 'fedproc':
            testacc = train_net_fedproc(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, global_centroids, current_round,
                                        device=device)
        elif args.alg == 'fedprox':
            testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device=device)
        elif args.alg == 'fedtemplate_add':
            testacc = train_net_template_add(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        elif args.alg == 'local_training':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        print("net %d, n_training: %d, final test acc %f" % (net_id, len(dataidxs), testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets

def local_train_net_scaffold(nets, global_model, c_nets, c_global, args, net_dataidx_map, train_dl=None, test_dl = None, device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    # c_global.to(device)
    # global_model.to(device)
    c_global.cuda()
    global_model.cuda()

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        train_dl_local = train_dl[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        # net.to(device)
        # c_nets[net_id].to(device)
        net.cuda()
        c_nets[net_id].cuda()
        n_epoch = args.epochs


        testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(dataidxs), testacc))
        avg_acc += testacc

    # c_global聚合
    for key in total_delta:
        total_delta[key] /= len(nets)
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(nets)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    print(now_time)
    print(args)

    dataset_logdir = os.path.join(args.logdir, args.dataset)
    mkdirs(dataset_logdir)

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % (now_time)
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(dataset_logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (now_time)
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(dataset_logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    if args.regression:
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_regression_data(
            args.dataset, args.datadir, dataset_logdir, args.partition, args.n_parties, beta=args.beta)
    else:
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
            args.dataset, args.datadir, dataset_logdir, args.partition, args.n_parties, beta=args.beta, n_niid_parties=args.n_niid_parties)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    if args.dataset != 'cinic10_val':
        train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                args.datadir,
                                                                                args.batch_size,
                                                                                args.batch_size)
    else:
        train_dl_global, val_dl, test_dl, train_ds_global, val_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                args.datadir,
                                                                                args.batch_size,
                                                                                args.batch_size)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    print(device)
    if args.alg=='fedfm' or args.alg=='fedproc':
        if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'cinic10', 'cinic10_val'}:
            n_classes = 10
        elif args.dataset == 'celeba':
            n_classes = 2
        elif args.dataset == 'cifar100':
            n_classes = 100
        elif args.dataset == 'tinyimagenet':
            n_classes = 200
        elif args.dataset == 'ham10000':
            n_classes = 7
        elif args.dataset == 'femnist':
            n_classes = 26
        elif args.dataset == 'emnist':
            n_classes = 47
        elif args.dataset == 'xray':
            n_classes = 2
        if args.regression:
            n_classes = 1
        if args.model.startswith('resnet18'):
            global_centroids = torch.zeros((n_classes, 512))
        elif args.model.startswith('resnet50'):
            global_centroids = torch.zeros((n_classes, 2048))
        global_centroids = global_centroids.cuda()
    train_local_dls=[]    
    val_local_dls=[]
    if args.dataset!='cinic10_val':
        for net_id, net in nets.items():
            dataidxs = net_dataidx_map[net_id]
            dataidxs_t = dataidxs[:int(0.8*len(dataidxs))]
            dataidxs_v = dataidxs[int(0.8*len(dataidxs)):]
            train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_t)
            train_local_dls.append(train_dl_local)
            if args.save_feature:
                val_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, 200, 32, dataidxs_v, drop_last=False)
            else:
                val_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_v, drop_last=False)
            val_local_dls.append(val_dl_local)
    else:
        for net_id, net in nets.items():
            dataidxs = net_dataidx_map[net_id]
            train_dl_local, _, _, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            train_local_dls.append(train_dl_local)
        val_local_dls.append(val_dl)

    best_acc=0
    best_test_acc=0
    test_acc_list = []
    acc_dir = os.path.join(dataset_logdir, 'acc_list')
    if not os.path.exists(acc_dir):
        os.mkdir(acc_dir)
    feature_dir = os.path.join(dataset_logdir, 'features')
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
    feature_dir = os.path.join(feature_dir, f'{now_time}')
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)

    acc_path = os.path.join(dataset_logdir, f'acc_list/{now_time}.npy')

    if args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                        
        for round in range(n_comm_rounds):
            print("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)



            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)



            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            print('global n_training: %d' % len(train_dl_global))
            print('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                print('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                print('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Train accuracy: %f' % val_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Best accuracy: %f' % best_test_acc)


            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')

    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            if args.distribution_aware != 'not' and args.measure_difference == 'only_iid':
                party_list_this_round = party_list_this_round[args.n_niid_parties:]
                if round==0:
                    print(party_list_this_round)

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device)

            # total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
            # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            if args.distribution_aware != 'not':
                distribution_difference = get_distribution_difference(traindata_cls_counts, participation_clients=party_list_this_round, metric=args.measure_difference)
                if args.distribution_aware == 'division':
                    total_normalizer = sum([fed_avg_freqs[r]/distribution_difference[r] for r in range(len(party_list_this_round))])
                    fed_avg_freqs = [fed_avg_freqs[r]/distribution_difference[r] / total_normalizer for r in range(len(party_list_this_round))]
                    if round==0:
                        print(fed_avg_freqs)


            # if args.aggregate_only_iid:     # haven't considered partial participation
            #     accumulate_coef = 0.0
            #     for net_id in range(len(party_list_this_round)):
            #         if net_id < args.n_niid_parties:
            #             continue
            #         elif net_id == args.n_niid_parties:
            #             accumulate_coef+=fed_avg_freqs[net_id]
            #         else:
            #             accumulate_coef+=fed_avg_freqs[net_id]
            #     if round==0:
            #         print(accumulate_coef)

            #     for net_id, net in enumerate(nets_this_round.values()):
            #         net_para = net.state_dict()
            #         if net_id < args.n_niid_parties:
            #             continue
            #         elif net_id == args.n_niid_parties:
            #             for key in net_para:
            #                 global_w[key] = net_para[key] * fed_avg_freqs[net_id] / accumulate_coef
            #         else:
            #             for key in net_para:
            #                 global_w[key] += net_para[key] * fed_avg_freqs[net_id] / accumulate_coef

            for net_id, net in enumerate(nets_this_round.values()):         # 这会不会有问题，net_id 在keys里会不会是乱序的
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')
           
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
    
    elif args.alg == 'fednova':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device)


            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            client_step = [(len(net_dataidx_map[r]) // args.batch_size) for r in party_list_this_round]
            tao_eff = 0.0
            for j in range(len(fed_avg_freqs)):
                tao_eff += fed_avg_freqs[j] * client_step[j]
            correct_term = 0.0
            for j in range(len(fed_avg_freqs)):
                correct_term += fed_avg_freqs[j] / client_step[j] * tao_eff
                
            for key in global_w.keys():
                global_w[key] = (1.0 - correct_term)*global_w[key]
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id] / client_step[net_id] * tao_eff


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')
           
            mkdirs(args.modeldir+'fednova/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fednova/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fednova/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg == 'fedfm':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device, current_round=round)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)

            #feature_matching
            global_centroids = global_centroids.cpu()
            local_centroids, local_distributions = get_client_centroids_info(global_model, dataloaders=train_local_dls, model_name=args.model, dataset_name=args.dataset, party_list_this_round=party_list_this_round)
            global_centroids = get_global_centroids(local_centroids, local_distributions, global_centroids, momentum=0.0, equally_average=args.fm_avg_anchor)
            global_centroids = global_centroids.cuda()

            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')
           
            mkdirs(args.modeldir+'fedfm/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedfm/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedfm/'+'localmodel0'+args.log_file_name+'.pth')
    
    elif args.alg == 'fedproc':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device, current_round=round)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)

            #feature_matching
            global_centroids = global_centroids.cpu()
            local_centroids, local_distributions = get_client_centroids_info(global_model, dataloaders=train_local_dls, model_name=args.model, dataset_name=args.dataset, party_list_this_round=party_list_this_round)
            global_centroids = get_global_centroids(local_centroids, local_distributions, global_centroids, momentum=0.0, equally_average=args.fm_avg_anchor)
            global_centroids = global_centroids.cuda()

            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')
           
            mkdirs(args.modeldir+'fedproc/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedproc/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedproc/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            # logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')

            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')

    elif args.alg == 'feddyn':
        alpha_coef = 1e-2
        n_par = len(get_mdl_params([global_model])[0])
        local_param_list = np.zeros((args.n_parties, n_par)).astype('float32')
        init_par_list=get_mdl_params([global_model], n_par)[0]
        clnt_params_list  = np.ones(args.n_parties).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
        avg_model = copy.deepcopy(global_model)
        # avg_model.load_state_dict(copy.deepcopy(dict(global_model.named_parameters())))

        all_model = copy.deepcopy(global_model)
        # all_model.load_state_dict(copy.deepcopy(dict(global_model.named_parameters())))

        cld_model = copy.deepcopy(global_model)
        # cld_model.load_state_dict(copy.deepcopy(dict(global_model.named_parameters())))
        cld_mdl_param = get_mdl_params([cld_model], n_par)[0]
        weight_list = np.asarray([len(net_dataidx_map[i]) for i in range(args.n_parties)])
        weight_list = weight_list / np.sum(weight_list) * args.n_parties
        for round in range(n_comm_rounds):
            print("round:",round)
            party_list_this_round = party_list_rounds[round]
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32).cuda()

            for clnt in party_list_this_round:
                # print('---- Training client %d' %clnt)
                train_dataloader=train_local_dls[clnt]
                model = copy.deepcopy(global_model).cuda()
                # Warm start from current avg model
                model.load_state_dict(cld_model.state_dict())
                for params in model.parameters():
                    params.requires_grad = True

                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt] # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32, device='cuda')
                loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=alpha_coef+args.reg)
                model.train()
                model.cuda()
                for e in range(args.epochs):
                    # Training
                    # epoch_loss_collector = []
                    for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                        batch_x = batch_x.cuda()
                        batch_y = batch_y.cuda()
                        
                        batch_x.requires_grad=False
                        batch_y.requires_grad=False
                        
                        optimizer.zero_grad()
                        _,_,y_pred = model(batch_x)
                        
                        ## Get f_i estimate 
                        loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
                        loss_f_i = loss_f_i / list(batch_y.size())[0]
                        
                        # Get linear penalty on the current parameter estimates
                        local_par_list = None
                        for param in model.parameters():
                            if not isinstance(local_par_list, torch.Tensor):
                            # Initially nothing to concatenate
                                local_par_list = param.reshape(-1)
                            else:
                                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                        loss_algo = alpha_coef_adpt * torch.sum(local_par_list * (-cld_mdl_param_tensor + local_param_list_curr))
                        loss = loss_f_i + loss_algo

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10) # Clip gradients
                        optimizer.step()
                    # print("epoch:",e," loss:",loss.item())
                
                # Freeze model
                for params in model.parameters():
                    params.requires_grad = False
                model.eval()
                curr_model_par = get_mdl_params([model], n_par)[0]

                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                local_param_list[clnt] += curr_model_par-cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

                # 输出一下每个client的acc
                model.cuda()
                test_acc, conf_matrix, _ = compute_accuracy(model, test_dl, get_confusion_matrix=True, device=device)
                # print("---AFTER LOCAL TRAINING---")
                print("Training network %s, n_training: %d, final test acc %f." % (str(clnt), len(train_dataloader.dataset), test_acc))
                # print("\n")
                model.to('cpu')

            avg_mdl_param = np.mean(clnt_params_list[party_list_this_round], axis = 0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

            avg_model = set_client_from_params(copy.deepcopy(global_model), avg_mdl_param)
            all_model = set_client_from_params(copy.deepcopy(global_model), np.mean(clnt_params_list, axis = 0))
            cld_model = set_client_from_params(copy.deepcopy(global_model), cld_mdl_param) 
        
            
            avg_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            if args.save_feature and (round+1)%10==0:
                save_features(model=avg_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(avg_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(avg_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)
            # test_result.append(test_acc)
            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                print('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Train accuracy: %f' % val_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')
           
            # mkdirs(args.modeldir+'fedavg/')
            avg_model.to('cpu')

    elif args.alg == 'scaffold':

        print("----SCAFFOLD----\n")
        logger.info("Initializing nets")

        # 初始化c_nets和c_global
        c_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
        c_globals, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        # 正片开始
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            print("In communication round:" + str(round))

            party_list_this_round = party_list_rounds[round]
            if args.distribution_aware != 'not' and args.measure_difference == 'only_iid':
                party_list_this_round = party_list_this_round[args.n_niid_parties:]
                if round==0:
                    print(party_list_this_round)
            if args.sample_fraction<1.0:
                print(f'Clients this round : {party_list_this_round}')

            # Model Initialization
            global_para = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_para)

            local_train_net_scaffold(nets_this_round, global_model, c_nets, c_global, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            # Model Aggregation
            for idx in range(len(party_list_this_round)):
                net_para = nets[party_list_this_round[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            # train_acc = compute_accuracy(global_model, train_dl_global)
            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')

            mkdirs(args.modeldir+'scaffold/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir+'scaffold/'+'globalmodel'+args.log_file_name+'.pth')
    
    elif args.alg == 'fedtemplate':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')
           
            mkdirs(args.modeldir+'fedtemplate/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedtemplate/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedtemplate/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg == 'fedtemplate_add':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            if args.save_feature and (round+1)%10==0:
                save_features(model=global_model, dataloaders=val_local_dls, save_dir=feature_dir, round=round)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            
            print(f'>> Round {round} test accuracy : {test_acc} | Best Acc : {best_test_acc}')
           
            mkdirs(args.modeldir+'fedtemplate/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedtemplate/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedtemplate/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg in ['fedopt_adagrad', 'fedopt_adam', 'fedopt_yogi']:
        global_flat = get_flat_params_from(global_model).detach()
        momentum = torch.zeros_like(global_flat).type_as(global_flat)
        v = (torch.ones_like(global_flat) * args.opt_tau).type_as(global_flat)
        optimizer_name = args.alg.split('_')[-1]
        
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            # Model updating
            global_flat = get_flat_params_from(global_model).detach()
            
            delta_w = torch.zeros_like(global_flat).type_as(global_flat)
            for net_id, net in enumerate(nets_this_round.values()):
                net_flat = get_flat_params_from(net).detach()
                delta_w = delta_w + (net_flat - global_flat) * fed_avg_freqs[net_id]
            
            if optimizer_name=='adagrad':
                opt_beta_1 = 0.0
                global_eta = 0.1
                momentum = opt_beta_1 * momentum + (1-opt_beta_1) * delta_w
                v = v + delta_w * delta_w
                # opt_global_lr = math.pow(10, -1.5)
                
            elif optimizer_name=='yogi':
                opt_beta_1 = 0.9
                opt_beta_2 = 0.99
                global_eta = 0.01
                momentum = opt_beta_1 * momentum + (1-opt_beta_1) * delta_w
                v -= (1-opt_beta_2) * delta_w * delta_w * torch.sign(v-delta_w * delta_w)
                # opt_global_lr = math.pow(10, -1.5)

            elif optimizer_name=='adam':
                opt_beta_1 = 0.9
                opt_beta_2 = 0.99
                global_eta = 0.01
                momentum = opt_beta_1 * momentum + (1-opt_beta_1) * delta_w
                v = opt_beta_2 * v + (1-opt_beta_2) * delta_w * delta_w
                # opt_global_lr = math.pow(10, -1.5)
            else:
                raise NotImplementedError
            
            global_flat = global_flat + global_eta * momentum / (torch.sqrt(v)+args.opt_tau)

            set_flat_params_to(global_model, global_flat)
            
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()

            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            print(f'Round {round} test accuracy : {test_acc}')
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
           
            mkdirs(args.modeldir+f'{args.alg}/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+f'{args.alg}/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+f'{args.alg}/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg in ['v2_fedopt_adagrad', 'v2_fedopt_adam', 'v2_fedopt_yogi']:
        pre_weights_np_list = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        m_t = None
        v_t = None
        optimizer_name = args.alg.split('_')[-1]
        
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            
            # Model updating
            aggregated_weights_np_list = get_aggregated_weights_np_list(nets_this_round, fed_avg_freqs)
            
            if optimizer_name=='adagrad':
                pre_weights_np_list, m_t, v_t = aggregation_adagrad(pre_weights_np_list=pre_weights_np_list, aggregated_weights_np_list=aggregated_weights_np_list, m_t=m_t, v_t=v_t)
            elif optimizer_name=='yogi':
                pre_weights_np_list, m_t, v_t = aggregation_yogi(pre_weights_np_list=pre_weights_np_list, aggregated_weights_np_list=aggregated_weights_np_list, m_t=m_t, v_t=v_t)
            elif optimizer_name=='adam':
                pre_weights_np_list, m_t, v_t = aggregation_adam(pre_weights_np_list=pre_weights_np_list, aggregated_weights_np_list=aggregated_weights_np_list, m_t=m_t, v_t=v_t)
            else:
                raise NotImplementedError
            
            set_model_using_np_list(net=global_model, weights_np_list=pre_weights_np_list)
            
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()

            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            print(f'>> Round {round} test accuracy : {test_acc}')
            test_acc_list.append(test_acc)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                logger.info('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                logger.info('>> Global Model Train accuracy: %f' % val_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
           
            mkdirs(args.modeldir+f'{args.alg}/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+f'{args.alg}/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+f'{args.alg}/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')
    
    print('>> Global Model Best accuracy: %f' % best_test_acc)
    print(args)
    print(f'>> Start time : {now_time} | End time : {datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")}')
    logger.info('>> Test ACC List: %s' % str(test_acc_list))
    np.save(acc_path, np.array(test_acc_list))