#!/bin/bash

# Model and context strategy constants
context_sample_strategy="random"
num_train_epochs=10  # May need to later adjust this if we aren't reaching convergence 
force_reload="false"   # Don't reload datasets when unnecessary 

# Models to loop over
model_names=("gpt2" "meta-llama/Llama-3.1-8B-Instruct")

# Hyperparameters for grid search
num_random_contexts_options=(1 3 5 10 15)
embedding_pool_strategies=("last" "mean")

# Loop over each model and hyperparameter combination
for model_name in "${model_names[@]}"; do
    echo "Starting grid search for model: $model_name"
    
    for num_random_contexts in "${num_random_contexts_options[@]}"; do
        for embedding_pool_strategy in "${embedding_pool_strategies[@]}"; do
            echo "Starting grid search with num_random_contexts=${num_random_contexts} and embedding_pool_strategy=${embedding_pool_strategy}"

            if [[ "$num_random_contexts" -eq 1 && "$embedding_pool_strategy" == "last" ]]; then
                echo "Skipping num_random_contexts=${num_random_contexts}, embedding_pool_strategy=${embedding_pool_strategy} as it already done."
                continue
            fi

            if [[ "$num_random_contexts" -eq 3 && "$embedding_pool_strategy" == "last" ]]; then
                echo "Skipping num_random_contexts=${num_random_contexts}, embedding_pool_strategy=${embedding_pool_strategy} as it already done."
                continue
            fi

            # Generate data and train
            if bash submit_job_chatbot_arena.sh "$model_name" "$context_sample_strategy" "$num_random_contexts" "$embedding_pool_strategy" "$num_train_epochs" "$force_reload"; then
                echo -e "\n✅ Successfully completed for model=${model_name}, num_random_contexts=${num_random_contexts}, embedding_pool_strategy=${embedding_pool_strategy}\n"
            else
                echo -e "\n🚨🚨🚨 WARNING: Failed for model=${model_name}, num_random_contexts=${num_random_contexts}, embedding_pool_strategy=${embedding_pool_strategy} 🚨🚨🚨\n"
            fi
        done
    done
done