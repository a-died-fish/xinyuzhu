CUDA_VISIBLE_DEVICES=1 python main_per.py --lam_fm 50.0 --start_ep_fm 20 --lr=0.01 --use_project_head 0 --dataset=cifar10 --device cuda:2 --model=resnet18_7 --alg=perfedfm --epochs=10 --comm_round=100 --n_parties=10 --partition=noniid --beta=0.5 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset'