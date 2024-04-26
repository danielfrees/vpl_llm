
# Set model_type to be 'gpt2' or 'llama' here
model_type='gpt2'


# Generate LLM embeddings for UltraFeedback dataset
subsets="helpfulness honesty instruction_following truthfulness"
for subset in ${subsets}
do
    python -m hidden_context.data_utils.data_processing --output_dir data/UltraFeedback_in_context_fixed/ \
    --data_path data/UltraFeedback --data_subset ${subset} --data_split test --model_type ${model_type} \
    --other_subsets ultra_feedback

    python -m hidden_context.data_utils.data_processing --output_dir data/UltraFeedback_in_context_fixed/ \
    --data_path data/UltraFeedback --data_subset ${subset} --data_split train --model_type ${model_type} \
    --other_subsets ultra_feedback
done
