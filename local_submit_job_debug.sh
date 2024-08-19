export WANDB_MODE=online
export WANDB_PROJECT=vpl
export WANDB_RUN_GROUP=$1
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Set model_name to be 'gpt2' or 'meta-llama/Llama-2-7b-hf' here
model_name='gpt2'

# Set scale of variance at vae_utils.py Line 117
# full: --controversial_only False --up_sampling False
# controversial: --controversial_only True
# up-sampling: --controversial_only False --up_sampling True

# Train VPL on full/controversial/up-sampling Pets dataset
#python -m hidden_context.train_llm_vae_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/simple_pets/gpt2" \
#        --num_train_epochs=10 \
#        --reward_model_type=vae \
#        --data_subset=both \
#        --log_dir="logs/gpt2_simple_pets" \
#        --bf16 True \
#        --fp16 False \
#        --per_device_train_batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --latent_dim 512 \
#        --hidden_dim 512 \
#        --learning_rate 1e-4 \
#        --use_annealing True \
#        --kl_loss_weight 1e-4 \
#        --fixed_contexts True \
#        --fixed_llm_embeddings False \
#        --use_last_token_embedding True \
#        --up_sampling True \
#        --controversial_only False \
#        --seed 0


# Train Dylan's models on full/controversial/up-sampling Pets dataset
#model_type=$2
#python -m hidden_context.train_llm_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/simple_pets/gpt2" \
#        --num_train_epochs=2 \
#        --reward_model_type=${model_type} \
#        --data_subset=both \
#        --log_dir="logs/gpt2_simple_pets" \
#        --bf16 True \
#        --fp16 False \
#        --per_device_train_batch_size 4 \
#         --gradient_accumulation_steps 8 \
#        --learning_rate 1e-4 \
#        --controversial_only False \
#        --up_sampling True \
#        --seed 0


augment_type="84"
#
# Train VPL on UltraFeedback two-user dataset
#python -m hidden_context.train_llm_vae_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/P_survey_100_variable/gpt2" \
#        --num_train_epochs=2 \
#        --reward_model_type=vae \
#        --data_subset=all \
#        --log_dir="logs/gpt2_P_survey_100_variable" \
#        --bf16 True \
#        --fp16 False \
#        --per_device_train_batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --latent_dim 512 \
#        --hidden_dim 512 \
#        --learning_rate 1e-4 \
#        --use_annealing True \
#        --kl_loss_weight 3e-6 \
#        --controversial_only True \
#        --fixed_contexts True \
#        --fixed_llm_embeddings False \
#        --up_sampling False \
#        --other_subsets ${augment_type} \
#        --use_last_token_embedding True \
#        --seed 0


# Train Dylan's models on UltraFeedback two-user dataset
#model_type=$2
#python -m hidden_context.train_llm_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/P_survey_100_variable/gpt2" \
#        --num_train_epochs=2 \
#        --reward_model_type=${model_type} \
#        --data_subset=all \
#        --log_dir="logs/gpt2_P_survey_100_variable" \
#        --bf16 True \
#        --fp16 False \
#        --per_device_train_batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --learning_rate 1e-4 \
#        --controversial_only True \
#        --up_sampling False \
#        --other_subsets ${augment_type} \
#        --seed 0


# Train VPL on UltraFeedback four-user dataset
python -m hidden_context.train_llm_vae_preference_model \
        --model_name=${model_name} \
        --data_path="data/P_4_survey_100/gpt2" \
        --num_train_epochs=2 \
        --reward_model_type=vae \
        --data_subset=all \
        --log_dir="logs/gpt2_P_4_survey_100" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --latent_dim 512 \
        --hidden_dim 512 \
        --learning_rate 1e-4 \
        --use_annealing True \
        --kl_loss_weight 3e-6 \
        --controversial_only True \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --up_sampling False \
        --other_subsets single \
        --use_last_token_embedding True \
        --seed 0

#model_type=$2
#python -m hidden_context.train_llm_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/P_4_survey_100/gpt2" \
#        --num_train_epochs=2 \
#        --reward_model_type=${model_type} \
#        --data_subset=all \
#        --log_dir="logs/gpt2_P_4_survey_100" \
#        --bf16 True \
#        --fp16 False \
#        --per_device_train_batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --learning_rate 1e-4 \
#        --controversial_only True \
#        --up_sampling False \
#        --other_subsets single \
#        --seed 0
