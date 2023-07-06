# sh fm_sh/fedproc.sh

# # CIFAR-10 / NIID-1
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --alg=fedproc \
# --save_feature 0 \
# --dataset=cifar10 \
# --model=resnet18_7 \
# --epochs=10 \
# --comm_round=100 \
# --n_parties=10 \
# --partition=noniid \
# --beta=0.5 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset'

# # CIFAR-10 / NIID-2
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --use_project_head 0 \
# --dataset=cifar10 \
# --device cuda:2 \
# --model=resnet18_7 \
# --alg=fedproc \
# --lr=0.01 \
# --epochs=10 \
# --comm_round=100 \
# --n_parties=10 \
# --partition=noniid-2 \
# --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset'

# # CINIC-10 / NIID-1
# CUDA_VISIBLE_DEVICES=3 python main.py \
# --print_local_test_acc 0 \
# --lr=0.01 \
# --use_project_head 0 \
# --dataset=cinic10 \
# --device cuda:2 \
# --model=resnet18 \
# --alg=fedproc \
# --epochs=10 \
# --comm_round=100 \
# --n_parties=10 \
# --partition=noniid \
# --beta=0.5 \
# --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/cinic10'

# # CINIC-10 / NIID-2
# CUDA_VISIBLE_DEVICES=3 python main.py \
# --print_local_test_acc 0 \
# --lr=0.01 \
# --use_project_head 0 \
# --dataset=cinic10 \
# --device cuda:2 \
# --model=resnet18 \
# --alg=fedproc \
# --epochs=10 \
# --comm_round=100 \
# --n_parties=10 \
# --partition=noniid-2 \
# --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/cinic10'

# # CIFAR-100 / NIID-1
# CUDA_VISIBLE_DEVICES=6 python main.py \
# --n_parties=10 \
# --use_project_head 0 \
# --dataset=cifar100 \
# --device cuda:2 \
# --model=resnet50 \
# --alg=fedproc \
# --lr=0.01 \
# --epochs=10 \
# --comm_round=100 \
# --partition=noniid \
# --beta=0.5 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/cifar100'

# # CIFAR-100 / NIID-2
# CUDA_VISIBLE_DEVICES=6 python main.py \
# --use_project_head 0 \
# --dataset=cifar100 \
# --device cuda:2 \
# --model=resnet50 \
# --alg=fedproc \
# --lr=0.01 \
# --epochs=10 \
# --comm_round=100 \
# --n_parties=10 \
# --partition=noniid-2 \
# --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/cifar100'