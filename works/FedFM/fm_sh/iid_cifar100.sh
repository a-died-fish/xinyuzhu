CUDA_VISIBLE_DEVICES=3 python main.py \
--alg=fedfm \
--mu 0.01 \
--lam_fm 50.0 \
--start_ep_fm 20 \
--use_project_head 0 \
--fm_avg_anchor 0 \
--dataset=cifar100 \
--device cuda:2 \
--model=resnet50 \
--lr=0.01 \
--epochs=10 \
--comm_round=100 \
--n_parties=10 \
--partition=iid \
--print_local_test_acc 0 \
--logdir='./logs/' --datadir='/GPFS/data/ruiye/fssl/dataset/cifar100'