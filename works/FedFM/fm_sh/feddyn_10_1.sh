CUDA_VISIBLE_DEVICES=4 python main.py --save_feature 0 --lr=0.01 --use_project_head 0 --dataset=cifar10 --device cuda:2 --model=resnet18_7_gn --alg=feddyn --epochs=10 --comm_round=100 --n_parties=10 --partition=noniid --beta=0.5 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset'