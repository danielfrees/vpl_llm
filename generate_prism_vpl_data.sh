#!/bin/bash

#model_type=${1:-"gpt2"}  
model_type=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
context_sample_strategy=${2:-"random"}  
num_random_contexts=${3:-5}  # Adjusted default to match function example
embedding_pool_strategy=${4:-"last"}

echo "Starting VPL data generation for PRISM with model type: ${model_type}"
echo "Context sample strategy: ${context_sample_strategy}"
echo "Num random contexts: ${num_random_contexts}"
echo "Embedding pool strategy: ${embedding_pool_strategy}"
echo "Synthetic dataset: False"

# Loop through each data split and run the data generation pipeline
for split in "train" "validation" "test"
do
    python -m vpl_modules.data_utils.generate_prism_data \
        --output_dir prism \
        --context_sample_strategy ${context_sample_strategy} \
        --num_random_contexts ${num_random_contexts} \
        --with_embeddings \
        --embedding_pool_strategy ${embedding_pool_strategy} \
        --model_type ${model_type} \
        --data_subset all \
        --data_split ${split}
done