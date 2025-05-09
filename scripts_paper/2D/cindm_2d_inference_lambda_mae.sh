#!/bin/bash

# Define the paths for inference_2d_guidance and logs
inference_2d_guidance_path="./saved/inference_2d_guidance_lambda_mae_new/"
log_path="./logs_guidance_lambda_mae_new"

# Define arrays for lambda values and num_boundaries
lambda_force_options=(1.0 10.0)
lambda_physics_options=(0.0 100.0)
num_boundaries_options=(2 3)

# Base command
base_cmd="python inference/inverse_design_2d.py \
    --ForceNet_path /data/01_cindm/checkpoint_path/force_surrogate_model.pth \
    --diffusion_model_path /data/01_cindm/checkpoint_path/diffusion_2d/ \
    --save_residuals True --inference_result_path $inference_2d_guidance_path \
    --num_batch 2"

# Create logs directory if it doesn't exist
mkdir -p "$log_path"

# Loop through all combinations
for num_boundaries in "${num_boundaries_options[@]}"; do
    for lambda_force in "${lambda_force_options[@]}"; do
        for lambda_physics in "${lambda_physics_options[@]}"; do
            # Construct the full command
            cmd="$base_cmd --num_boundaries $num_boundaries --lambda_force $lambda_force --lambda_physics $lambda_physics"
            
            # Create a unique log file name
            log_file="$log_path/run_boundaries${num_boundaries}_lambdaForce${lambda_force}_lambdaPhysics${lambda_physics}.log"
            
            # Run the command and redirect output to log file
            echo "Running: $cmd"
            echo "Logging output to: $log_file"
            eval $cmd > "$log_file" 2>&1
            
            echo "Finished run with num_boundaries=$num_boundaries, lambda_force=$lambda_force, and lambda_physics=$lambda_physics"
            echo "----------------------------------------"
        done
    done
done

echo "All runs completed."