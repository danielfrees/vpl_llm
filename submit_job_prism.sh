#!/bin/bash

export WANDB_MODE=online
export WANDB_PROJECT=vpl
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Argument signature, with default values
model_name=${1:-"gpt2"}
context_sample_strategy=${2:-"random"}
num_random_contexts=${3:-5}  
embedding_pool_strategy=${4:-"last"}
num_train_epochs=${5:-2}
force_reload=${6:-"false"}

DATA_PATH="data/prism/${model_name}/context_${context_sample_strategy}/numrandom_${num_random_contexts}/pooling_${embedding_pool_strategy}"

echo "Model name: ${model_name}"
echo "Context sample strategy: ${context_sample_strategy}"
echo "Num random contexts: ${num_random_contexts}"
echo "Embedding pool strategy: ${embedding_pool_strategy}"
echo "Num train epochs: ${num_train_epochs}"
echo "Force reload data: ${force_reload}"

# Check if data needs generation
if [[ "$force_reload" == "true" || ! -f "${DATA_PATH}/train.jsonl" || ! -f "${DATA_PATH}/validation.jsonl" || ! -f "${DATA_PATH}/test.jsonl" ]]; then
    echo "Generating VPL data for PRISM..."
    # Run the data generation script with the specified arguments
    bash generate_prism_vpl_data.sh ${model_name} ${context_sample_strategy} ${num_random_contexts} ${embedding_pool_strategy}
else
    echo "VPL data for PRISM already exists. Skipping data generation."
fi

# Train the model
python -m hidden_context.train_llm_vae_preference_model \
    --model_name=${model_name} \
    --data_path=${DATA_PATH} \
    --context_sample_strategy=${context_sample_strategy} \
    --num_random_contexts=${num_random_contexts} \
    --embedding_pool_strategy=${embedding_pool_strategy} \
    --num_train_epochs=${num_train_epochs} \
    --reward_model_type=vae \
    --data_subset=both \
    --log_dir="logs/${model_name}_prism" \
    --bf16 True \
    --fp16 False \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --latent_dim 512 \
    --hidden_dim 512 \
    --learning_rate 3e-4 \
    --use_annealing True \
    --kl_loss_weight 1e-4 \
    --fixed_contexts True \
    --fixed_llm_embeddings False \
    --use_last_token_embedding True \
    --up_sampling False \
    --controversial_only False \
    --seed 0