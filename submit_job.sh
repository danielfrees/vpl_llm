#!/bin/bash

#SBATCH --job-name=dpl
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sriyash@cs.washington.edu

#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

#SBATCH --chdir=/gscratch/weirdlab/sriyash/hidden-context
#SBATCH --export=all
#SBATCH --output=slurm/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm/%j-err.txt    # where STDERR goes

HOME_DIR=/gscratch/weirdlab/sriyash/hidden-context
export WANDB_MODE=online
export WANDB_PROJECT=dpl-baselines

source ${HOME}/.bashrc
conda activate dpl
cd $HOME_DIR


data_subset=$1
epochs=$2
model_type=$3

python -m hidden_context.train_llm_preference_model \
        --model_name="gpt2" \
        --num_train_epochs=$epochs \
        --reward_model_type=$model_type \
        --data_subset=$data_subset \
        --log_dir="data/gpt2_results" \


