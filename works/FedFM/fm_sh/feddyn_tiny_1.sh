CUDA_VISIBLE_DEVICES=7 python main.py --use_project_head 0 --dataset=tinyimagenet --device cuda:2 --model=resnet18_7_gn --alg=feddyn --lr=0.01 --epochs=10 --comm_round=40 --n_parties=10 --partition=noniid --beta=0.5 --logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/tiny_imagenet/tiny-imagenet-200'