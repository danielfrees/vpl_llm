
# Set model_type to be 'gpt2' or 'llama' here
model_type="gpt2"

# Generate Pets dataset
 python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets_noemb/${model_type} \
 --data_path data/relabeled_hh_rlhf --with_embeddings False --synthetic_dataset True \
 --model_type ${model_type} --data_subset helpful --data_split test --dataset_size 200

 python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets_noemb/${model_type} \
 --data_path data/relabeled_hh_rlhf --with_embeddings False --synthetic_dataset True \
 --model_type ${model_type} --data_subset helpful --data_split train --dataset_size 2000

 python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets_noemb/${model_type} \
 --data_path data/relabeled_hh_rlhf --with_embeddings False --synthetic_dataset True \
 --model_type ${model_type} --data_subset harmless --data_split test --dataset_size 200

 python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets_noemb/${model_type} \
 --data_path data/relabeled_hh_rlhf --with_embeddings False --synthetic_dataset True \
 --model_type ${model_type} --data_subset harmless --data_split train --dataset_size 2000
