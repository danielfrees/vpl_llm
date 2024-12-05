#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a6000:1
#SBATCH --job-name=gridsearch
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --output=gridsearch-%j.out
#SBATCH --error=gridsearch-%j.err
#SBATCH --partition=jag-standard
#SBATCH --time=14-0
#SBATCH --nodes=1

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh ; conda activate icrm

export HF_HOME=/nlp/scr2/nlp/personal-rm/hf_cache/

cd /juice2/scr2/nlp/personal-rm/personalized-reward-models/personalized-reward-models/personalized_reward_models/vpl/vpl_llm

./gridsearch_prism.sh