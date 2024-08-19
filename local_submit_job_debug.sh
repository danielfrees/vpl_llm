export WANDB_MODE=online
export WANDB_PROJECT=vpl-temp
export WANDB_RUN_GROUP=$1
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
# python -m hidden_context.train_llm_vae_preference_model \
#         --model_name=${model_name} \
#         --data_path="data/simple_pets_balanced/gpt2" \
#         --num_train_epochs=100 \
#         --reward_model_type=vae \
#         --data_subset=both \
#         --log_dir="logs/gpt2_simple_pets_balanced_contrastive" \
#         --bf16 True \
#         --fp16 False \
#         --per_device_train_batch_size 4 \
#         --gradient_accumulation_steps 8 \
#         --latent_dim 512 \
#         --hidden_dim 512 \
#         --learning_rate 3e-4 \
#         --use_annealing True \
#         --kl_loss_weight 1e-6 \
#         --seed 0 \
#         --controversial_only True \
#         --fixed_contexts True \
#         --fixed_llm_embeddings False \
#         --use_causal_lm False \
#         --use_last_token_embedding True \
#         --use_attention_layer False \
#         --up_sampling False



# Train Dylan's models on full/controversial/up-sampling Pets dataset
model_type=$2
#python -m hidden_context.train_llm_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/simple_pets_last/gpt2" \
#        --num_train_epochs=2 \
#        --reward_model_type=${model_type} \
#        --data_subset=both \
#        --log_dir="logs/gpt2_simple_pets_last" \
#        --bf16 True \
#        --fp16 False \
#        --learning_rate 1e-4 \
#        --controversial_only False \
#        --up_sampling True \
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

augment_type="84"
##
## Train VPL on UltraFeedback dataset
#python -m hidden_context.train_llm_vae_preference_model \
#       --model_name=${model_name} \
#       --data_path="data/P_survey_100_variable/gpt2" \
#       --num_train_epochs=2 \
#       --reward_model_type=vae \
#       --data_subset=all \
#       --log_dir="logs/variable_survey_100" \
#       --bf16 True \
#       --fp16 False \
#       --per_device_train_batch_size 4 \
#       --gradient_accumulation_steps 8 \
#       --latent_dim 512 \
#       --hidden_dim 512 \
#       --learning_rate 1e-4 \
#       --use_annealing True \
#       --kl_loss_weight 3e-6 \
#       --controversial_only True \
#       --fixed_contexts True \
#       --fixed_llm_embeddings False \
#       --use_causal_lm False \
#       --use_attention_layer False \
#       --up_sampling False \
#       --other_subsets ${augment_type} \
#       --use_last_token_embedding True \
#       --use_transformer False \
#       --concat_chosen_rejected False \
#       --llm2vec False \
#       --llama_embeddings False \
#       --seed 0 \
        # --one_user="8"

python -m hidden_context.train_llm_vae_preference_model \
	--model_name=${model_name} \
	--data_path="data/P_4_survey_100_final/gpt2" \
	--num_train_epochs=2 \
	--reward_model_type=vae \
	--data_subset=all \
	--log_dir="logs/P_4_survey_100_final_contrastive" \
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
	--use_causal_lm False \
	--use_attention_layer False \
	--up_sampling False \
	--other_subsets single \
	--use_last_token_embedding True \
	--use_transformer False \
	--concat_chosen_rejected False \
	--llm2vec False \
	--llama_embeddings False \
	--seed 0 \
#	--one_user="8"

#model_type=$3
#python -m hidden_context.train_llm_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/P_4_survey_100_final/gpt2" \
#        --num_train_epochs=2 \
#        --reward_model_type=${model_type} \
#        --data_subset=all \
#        --log_dir="logs/P_4_survey_100_final_single_mode" \
#        --bf16 True \
#        --fp16 False \
#        --per_device_train_batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --learning_rate 1e-4 \
#        --controversial_only True \
#        --up_sampling False \
#        --other_subsets single \
#        --seed 0 \
#	--one_user="8"

# Train Dylan's models on UltraFeedback dataset
#model_type=$2
#python -m hidden_context.train_llm_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/large_survey_100/gpt2" \
#        --num_train_epochs=2 \
#        --reward_model_type=${model_type} \
#        --data_subset=all \
#        --log_dir="logs/final_large_survey_100_0" \
#        --bf16 True \
#        --fp16 False \
#        --per_device_train_batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --learning_rate 3e-4 \
#        --controversial_only True \
#        --up_sampling False \
#        --other_subsets ${augment_type} \
#        --seed 2

#
#python -m hidden_context.train_llm_vae_preference_model \
#        --model_name=${model_name} \
#        --data_path="data/llm2vec_survey" \
#        --num_train_epochs=2 \
#        --reward_model_type=vae \
#        --data_subset=all \
#        --log_dir="logs/llm2vec_survey" \
#        --bf16 True \
#        --fp16 False \
#        --per_device_train_batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --latent_dim 512 \
#        --hidden_dim 512 \
#        --learning_rate 3e-4 \
#        --use_annealing True \
#        --kl_loss_weight 1e-4 \
#        --controversial_only True \
#        --fixed_contexts True \
#        --fixed_llm_embeddings False \
#        --use_causal_lm False \
#        --use_attention_layer False \
#        --up_sampling False \
#        --other_subsets ${augment_type} \
#        --use_last_token_embedding True \
#        --use_transformer False \
#        --concat_chosen_rejected False \
#        --llm2vec False \
#        --seed 0 \
