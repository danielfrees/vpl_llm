# !/bin/bash


# ckpts=("base_gpt2__0_3e-06_cosine_1_peft_last_checkpoint" \
#        "categorical_gpt2__0_3e-06_cosine_1_10_0.1_peft_last_checkpoint" \
#        "mean_and_variance_gpt2__0_3e-06_cosine_1_0.0_peft_last_checkpoint" \
#     )
# num_outputs=(1 10 2)
# models=("base" "categorical" "mean_and_variance")

# for index in ${!ckpts[*]}
# do
#     python -m hidden_context.evaluate_llm_preference_model \
#         --model_name=gpt2 \
#         --num_outputs=${num_outputs[$index]} \
#         --reward_model_checkpoint="/gscratch/weirdlab/sriyash/hidden-context/saved_gpt2_models_relabelled/${ckpts[$index]}"

#     python -m hidden_context.evaluate_assistant_responses \
#         --input=data/jailbroken_responses.jsonl \
#         --model_name=gpt2 \
#         --num_outputs=${num_outputs[$index]} \
#         --reward_model_checkpoints "/gscratch/weirdlab/sriyash/hidden-context/saved_gpt2_models_relabelled/${ckpts[$index]}" \
#         --reward_model_names ${models[$index]} \
#         --output /gscratch/weirdlab/sriyash/hidden-context/saved_gpt2_models_relabelled/${ckpts[$index]}/jailbroken_responses.jsonl
# done

python -m hidden_context.evaluate_llm_vae_preference_model \
        --model_name=gpt2 \
        --reward_model_checkpoint=saved_gpt2_models_relabelled/vae_gpt2__0_3e-06_cosine_1_0.01_512_1024_peft_last_checkpoint/
        
        # --num_outputs=1 \
        # --reward_model_checkpoint="/gscratch/weirdlab/sriyash/hidden-context/data/gpt2_results/${ckpt}"

python -m hidden_context.evaluate_assistant_responses_vae \
    --input=data/jailbroken_responses.jsonl \
    --model_name=gpt2 \
    --reward_model_checkpoint=saved_gpt2_models_relabelled/vae_gpt2__0_3e-06_cosine_1_0.01_512_1024_peft_last_checkpoint/ \
    --output=saved_gpt2_models_relabelled/vae_gpt2__0_3e-06_cosine_1_0.01_512_1024_peft_last_checkpoint/jailbroken_responses.jsonl

#     --num_outputs=1 \
#     --reward_model_checkpoints PATH_1/TO/last_checkpoint PATH_2/TO/last_checkpoint \
#     --reward_model_names model_1 model_2 \
#     --output PATH/TO/output.jsonl