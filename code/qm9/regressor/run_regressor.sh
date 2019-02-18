#!bin/bash

python train_regressor.py  -regressor_saved_dir  './regressor/regressor_pretrained_pretrained_vae_output' -vanilla_vae_save_dir  './../vanilla_supervised_vae' -training_data_dir '../data/data_100'
