CUDA_VISIBLE_DEVICES=2 python main.py --n_parties=10 --use_project_head 0 --dataset=cifar100 --device cuda:2 --model=resnet50 --alg=fednova --lr=0.01 --epochs=10 --comm_round=100 --partition=noniid-2 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/cifar100'