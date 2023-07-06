CUDA_VISIBLE_DEVICES=3 python main.py \
--alg=fedavg \
--save_feature 0 \
--dataset=cifar10 \
--model=resnet18_7 \
--epochs=10 \
--comm_round=100 \
--n_parties=10 \
--partition=noniid \
--beta=0.5 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset'
# sh fm_sh/fedavg_10_1.sh