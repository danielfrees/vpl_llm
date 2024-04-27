export WANDB_MODE=online
export WANDB_PROJECT=vpl-debug
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# export CUDA_VISIBLE_DEVICES=0

# Set model_name to be 'gpt2' or 'meta-llama/Llama-2-7b-hf' here
model_name='gpt2'

# Set scale of variance at vae_utils.py Line 117
# full: --controversial_only False --up_sampling False
# controversial: --controversial_only True
# up-sampling: --controversial_only False --up_sampling True

# Train VPL on full/controversial/up-sampling Pets dataset
#python -m hidden_context.train_llm_vae_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/simple_pets_0_01/gpt2" \
#        --num_train_epochs=1 \
#        --reward_model_type=vae \
#        --data_subset=both \
#        --log_dir="logs/gpt2_simple_pets_0_01" \
#        --bf16 True \
#        --fp16 False \
#        --learning_rate 1e-4 \
#        --use_annealing True \
#        --kl_loss_weight 1e-6 \
#        --controversial_only False \
#        --fixed_contexts True \
#        --fixed_llm_embeddings False \
#        --use_causal_lm False \
#        --up_sampling False \
#        --seed 0


# Train Dylan's models on full/controversial/up-sampling Pets dataset
#model_type=$1
#python -m hidden_context.train_llm_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/simple_pets_0_01/gpt2" \
#        --num_train_epochs=1 \
#        --reward_model_type=${model_type} \
#        --data_subset=both \
#        --log_dir="logs/gpt2_simple_pets_0_01" \
#        --bf16 True \
#        --fp16 False \
#        --learning_rate 1e-4 \
#        --controversial_only False \
#        --up_sampling False \
#        --seed 0


# Train VPL on HH-RLHF dataset
#python -m hidden_context.train_llm_vae_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/relabeled_hh_rlhf_in_context_fixed/gpt2" \
#        --num_train_epochs=1 \
#        --reward_model_type=vae \
#        --data_subset=both \
#        --log_dir="logs/gpt2_relabeled_hh_rlhf" \
#        --bf16 True \
#        --fp16 False \
#        --learning_rate 1e-4 \
#        --use_annealing True \
#        --kl_loss_weight 1e-6 \
#        --controversial_only False \
#        --fixed_contexts True \
#        --fixed_llm_embeddings False \
#        --use_causal_lm False \
#        --up_sampling False \
#        --seed 0


# Train Dylan's models on HH-RLHF dataset
#model_type=$1
#python -m hidden_context.train_llm_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/relabeled_hh_rlhf_in_context_fixed/gpt2" \
#        --num_train_epochs=1 \
#        --reward_model_type=${model_type} \
#        --data_subset=both \
#        --log_dir="logs/gpt2_relabeled_hh_rlhf" \
#        --bf16 True \
#        --fp16 False \
#        --learning_rate 1e-4 \
#        --controversial_only False \
#        --up_sampling False \
#        --seed 0


# Train VPL on UltraFeedback dataset
#python -m hidden_context.train_llm_vae_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/UltraFeedback_in_context_fixed/gpt2" \
#        --num_train_epochs=1 \
#        --reward_model_type=vae \
#        --data_subset=all \
#        --log_dir="logs/gpt2_UltraFeedback" \
#        --bf16 True \
#        --fp16 False \
#        --learning_rate 1e-4 \
#        --use_annealing True \
#        --kl_loss_weight 1e-6 \
#        --controversial_only False \
#        --fixed_contexts True \
#        --fixed_llm_embeddings False \
#        --use_causal_lm False \
#        --up_sampling False \
#        --other_subsets ultra_feedback \
#        --seed 0


# Train Dylan's models on UltraFeedback dataset
#model_type=$1
#python -m hidden_context.train_llm_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/UltraFeedback_in_context_fixed/gpt2" \
#        --num_train_epochs=1 \
#        --reward_model_type=${model_type} \
#        --data_subset=all \
#        --log_dir="logs/gpt2_UltraFeedback" \
#        --bf16 True \
#        --fp16 False \
#        --learning_rate 1e-4 \
#        --controversial_only False \
#        --up_sampling False \
#        --other_subsets ultra_feedback \
#        --seed 0


# Train VPL on UltraFeedback dataset
#python -m hidden_context.train_llm_vae_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/UltraFeedback_pos_neg_in_context_fixed/gpt2" \
#        --num_train_epochs=1 \
#        --reward_model_type=vae \
#        --data_subset=all \
#        --log_dir="logs/gpt2_UltraFeedback_pos_neg" \
#        --bf16 True \
#        --fp16 False \
#        --learning_rate 1e-4 \
#        --use_annealing True \
#        --kl_loss_weight 1e-6 \
#        --controversial_only False \
#        --fixed_contexts True \
#        --fixed_llm_embeddings False \
#        --use_causal_lm False \
#        --up_sampling False \
#        --other_subsets pos_neg \
#        --seed 0


# Train Dylan's models on UltraFeedback dataset
model_type=$1
python -m hidden_context.train_llm_preference_model \
        --model_name=${model_name} \
        --data_path="data/UltraFeedback_pos_neg_in_context_fixed/gpt2" \
        --num_train_epochs=1 \
        --reward_model_type=${model_type} \
        --data_subset=all \
        --log_dir="logs/gpt2_UltraFeedback_pos_neg" \
        --bf16 True \
        --fp16 False \
        --learning_rate 1e-4 \
        --controversial_only False \
        --up_sampling False \
        --other_subsets pos_neg \
        --seed 0
