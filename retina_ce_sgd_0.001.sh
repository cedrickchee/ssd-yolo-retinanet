CUDA_VISIBLE_DEVICES=0 python trainer.py --experiment_name retina_ce_sgd_0.001 --decay_steps 600000 1000000 1400000 --means 104 117 123 --lr 1e-3 --loss_type 'ce' --optim 'SGD' --batch_size 5
