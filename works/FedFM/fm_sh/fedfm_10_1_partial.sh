CUDA_VISIBLE_DEVICES=4 python main.py --model=resnet18_7 --n_parties=100 --sample_fraction 0.1 --beta=0.5 --dataset=cifar10 --alg=fedfm --epochs=10 --comm_round=100 --partition=noniid --use_project_head 0 --device cuda:2 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset'