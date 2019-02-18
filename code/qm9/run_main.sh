#!/bin/bash

python main.py -RL_pram 50 -vanilla_vae_save_dir './vanilla_supervised_vae' -training_data_dir './data/data_100' -regressor_saved_dir './regressor/regressor_pretrained_pretrained_vae_output' -num_epochs 500
