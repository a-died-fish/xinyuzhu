CUDA_VISIBLE_DEVICES=0 python main.py --partition=noniid-3-5 --lam_fm 50.0 --start_ep_fm 20 --use_project_head 0 --dataset=cifar10 --device cuda:2 --model=resnet18_7 --alg=fedfm --lr=0.01 --epochs=10 --comm_round=100 --n_parties=10 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset'