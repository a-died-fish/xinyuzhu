CUDA_VISIBLE_DEVICES=6 python main.py --print_local_test_acc 0 --lr=0.01 --use_project_head 0 --dataset=cinic10 --device cuda:2 --model=resnet18 --alg=fednova --epochs=10 --comm_round=100 --n_parties=10 --partition=noniid-2 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/cinic10'