
# Set model_type to be 'gpt2' or 'llama' here
model_type='gpt2'

# Generate Pets dataset
# python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets_0_01/ \
# --data_path data/relabeled_hh_rlhf --with_embeddings True --add_controversial True --synthetic_dataset True \
# --use_causal_lm False --model_type ${model_type} --data_subset helpful --data_split test --dataset_size 200

# python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets_0_01/ \
# --data_path data/relabeled_hh_rlhf --with_embeddings True --add_controversial True --synthetic_dataset True \
# --use_causal_lm False --model_type ${model_type} --data_subset helpful --data_split train --dataset_size 2000

# python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets_0_01/ \
# --data_path data/relabeled_hh_rlhf --with_embeddings True --add_controversial True --synthetic_dataset True \
# --use_causal_lm False --model_type ${model_type} --data_subset harmless --data_split test --dataset_size 200

# python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets_0_01/ \
# --data_path data/relabeled_hh_rlhf --with_embeddings True --add_controversial True --synthetic_dataset True \
# --use_causal_lm False --model_type ${model_type} --data_subset harmless --data_split train --dataset_size 2000


# Generate LLM embeddings for HH-RLHF dataset
# python -m hidden_context.data_utils.data_processing --output_dir data/relabeled_hh_rlhf_in_context_fixed/ \
# --data_path data/relabeled_hh_rlhf --data_subset helpful --data_split test --model_type ${model_type}
#
# python -m hidden_context.data_utils.data_processing --output_dir data/relabeled_hh_rlhf_in_context_fixed/ \
# --data_path data/relabeled_hh_rlhf --data_subset helpful --data_split train --model_type ${model_type}
#
# python -m hidden_context.data_utils.data_processing --output_dir data/relabeled_hh_rlhf_in_context_fixed/ \
# --data_path data/relabeled_hh_rlhf --data_subset harmless --data_split test --model_type ${model_type}
#
# python -m hidden_context.data_utils.data_processing --output_dir data/relabeled_hh_rlhf_in_context_fixed/ \
# --data_path data/relabeled_hh_rlhf --data_subset harmless --data_split train --model_type ${model_type}