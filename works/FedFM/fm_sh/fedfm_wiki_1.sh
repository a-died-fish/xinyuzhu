CUDA_VISIBLE_DEVICES=7 python main_regression.py --num_anchor 50 --regression --beta=5.0 --use_project_head 1 --lam_fm 10.0 --start_ep_fm 20 --dataset=wiki --device cuda:2 --model=resnet18 --alg=fedfm --lr=0.01 --epochs=10 --comm_round=100 --n_parties=10 --partition=noniid --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/wiki_crop'