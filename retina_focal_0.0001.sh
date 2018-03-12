CUDA_VISIBLE_DEVICES=0 python3 trainer.py --experiment_name retina_focal_sgd_.0001 --decay_steps 600000 1000000 1400000 --means 104 117 123 --lr 1e-4 --optim 'SGD'
