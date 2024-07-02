#! /bin/bash

# start ml_optim env
source ~/libs/env/start_ml_optim.sh

python inference/inverse_design_2d.py --ForceNet_path "/home/t-isaacju/data/01_cindm/checkpoint_path.bak/checkpoint_path/force_surrogate_model.pth" --diffusion_model_path "/home/t-isaacju/data/01_cindm/checkpoint_path.bak/checkpoint_path/diffusion_2d/"