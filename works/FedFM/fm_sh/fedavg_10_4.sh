CUDA_VISIBLE_DEVICES=2 python main.py --distribution_aware='not' --measure_difference='cosine' --n_parties=11 --n_niid_parties=5 --lr=0.01 --use_project_head 0 --dataset=cifar10 --device cuda:2 --model=simple-cnn --alg=fedavg --epochs=10 --comm_round=100 --partition=noniid-4 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/cifar10'