#!/bin/bash

# Model and context strategy constants
model_name="meta-llama/Llama-3.1-8B-Instruct" # add llama3.1-8b later once implemented and tested
context_sample_strategy="random"
num_train_epochs = 10  # May need to later adjust this if we aren't reaching convergence 
force_reload="false"   # Don't reload datasets when unecessary 

# Hyperparameters for grid search
num_random_contexts_options=(1 3 5)
embedding_pool_strategies=("last")

# Loop over each combination
for num_random_contexts in "${num_random_contexts_options[@]}"; do
    for embedding_pool_strategy in "${embedding_pool_strategies[@]}"; do
        echo "Starting grid search with num_random_contexts=${num_random_contexts} and embedding_pool_strategy=${embedding_pool_strategy}"

        # Generate data and train
        if bash submit_job_prism.sh "$model_name" "$context_sample_strategy" "$num_random_contexts" "$embedding_pool_strategy" "$num_train_epochs" "$force_reload"; then
            echo -e "\n✅ Successfully completed for num_random_contexts=${num_random_contexts}, embedding_pool_strategy=${embedding_pool_strategy}\n"
        else
            echo -e "\n🚨🚨🚨 WARNING: Failed for num_random_contexts=${num_random_contexts}, embedding_pool_strategy=${embedding_pool_strategy} 🚨🚨🚨\n"
        fi
    done
done

echo "Grid search completed."