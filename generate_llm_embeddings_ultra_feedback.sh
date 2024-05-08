
# Set model_type to be 'gpt2' or 'llama' for model_type
# Set other_subsets to be 'ultra_feedback', 'pos_neg', 'set', or 'single'
model_type=$1
other_subsets=$2

# Generate LLM embeddings for UltraFeedback dataset
if [ "${other_subsets}" = "ultra_feedback" ]; then
    subsets="helpfulness honesty instruction_following truthfulness"
    postfix=""
elif [ "${other_subsets}" = "pos_neg" ]; then
    subsets="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    postfix="_pos_neg"
elif [ "${other_subsets}" = "set" ]; then
    subsets="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    postfix="_subset"
elif [ "${other_subsets}" = "single" ]; then
    subsets="8 4 2 1"
    postfix="_single"
else
    echo "Invalid!"
fi

echo "${subsets}"
#
for subset in ${subsets}
do
    python -m hidden_context.data_utils.data_processing --output_dir "data/UltraFeedback${postfix}_in_context_fixed_finegrained_filtered_CLS/" \
    --data_path "data/UltraFeedback${postfix}_finegrained_filtered" --data_subset ${subset} --data_split test --model_type ${model_type} \
    --other_subsets ${other_subsets} --add_controversial True

    python -m hidden_context.data_utils.data_processing --output_dir "data/UltraFeedback${postfix}_in_context_fixed_finegrained_filtered_CLS/" \
    --data_path "data/UltraFeedback${postfix}_finegrained_filtered" --data_subset ${subset} --data_split train --model_type ${model_type} \
    --other_subsets ${other_subsets} --add_controversial True
done
