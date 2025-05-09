#!/bin/bash

# Define the paths for inference_2d_guidance and logs
inference_2d_guidance_path="./saved/inference_2d_guidance_new/"
log_path="./logs_guidance_new"

# Define arrays for use_physics, physics_lambda, and num_boundaries
use_physics_options=(True False)
physics_lambda_options=(0.1 1.0)
num_boundaries_options=(1 2 3)

# Base command
base_cmd="python inference/inverse_design_2d.py \
    --ForceNet_path /data/01_cindm/checkpoint_path/force_surrogate_model.pth \
    --diffusion_model_path /data/01_cindm/checkpoint_path/diffusion_2d/ \
    --save_residuals True --inference_result_path $inference_2d_guidance_path \
    --num_batch 2"

# Create logs directory if it doesn't exist
mkdir -p "$log_path"

# Loop through all combinations
for use_physics in "${use_physics_options[@]}"; do
    for num_boundaries in "${num_boundaries_options[@]}"; do
        if [ "$use_physics" = "True" ]; then
            for lambda_physics in "${physics_lambda_options[@]}"; do
                # Construct the full command
                cmd="$base_cmd --use_physics_loss True --num_boundaries $num_boundaries --lambda_physics $lambda_physics"
                
                # Create a unique log file name
                log_file="$log_path/run_physics${use_physics}_boundaries${num_boundaries}_lambda${lambda_physics}.log"
                
                # Run the command and redirect output to log file
                echo "Running: $cmd"
                echo "Logging output to: $log_file"
                eval $cmd > "$log_file" 2>&1
                
                echo "Finished run with use_physics=$use_physics, num_boundaries=$num_boundaries, and lambda_physics=$lambda_physics"
                echo "----------------------------------------"
            done
        else
            # Construct the command without physics options
            cmd="$base_cmd --num_boundaries $num_boundaries"
            
            # Create a unique log file name
            log_file="$log_path/run_physics${use_physics}_boundaries${num_boundaries}.log"
            
            # Run the command and redirect output to log file
            echo "Running: $cmd"
            echo "Logging output to: $log_file"
            eval $cmd > "$log_file" 2>&1
            
            echo "Finished run with use_physics=$use_physics and num_boundaries=$num_boundaries"
            echo "----------------------------------------"
        fi
    done
done

echo "All runs completed."