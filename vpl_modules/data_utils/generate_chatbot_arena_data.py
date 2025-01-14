""" 
User-stratified data augmenation (reformatting, context augmentation, embeddings, etc.) 
for input into VPL training with cached LLM embeddings. 

CHATBOT ARENA
"""

import pandas as pd 
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
import os
from typing import List, Dict, Union, Tuple
from dotenv import load_dotenv
import argparse
from data_processing import generate_vpl_data
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
CHATBOT_ARENA_DATASET = 'MichaelR207/chatbot_arena_personalized_1023'

def main():
    parser = argparse.ArgumentParser(description="Generate PRISM data for VPL training with cached LLM embeddings")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory name")
    parser.add_argument("--model_type", type=str, default="gpt2", choices=["gpt2", "llama"], help="Model type for generating embeddings")
    parser.add_argument("--context_sample_strategy", type=str, default="random", choices=["random", "top2", "bestworst", "max", "cum"], help="Strategy for sampling context pairs")
    parser.add_argument("--num_random_contexts", type=int, default=5, help="Number of random context pairs to sample")
    parser.add_argument("--with_embeddings", action="store_true", help="Flag to generate embeddings and cache them ahead of time for faster VPL training (and inference).")
    parser.add_argument("--embedding_pool_strategy", type=str, default="last", choices=["mean", "last"], help="Pooling strategy for generating embeddings for input strings from the LLM. 'last' recommended for autoregressive models.")
    parser.add_argument("--data_subset", type=str, default="all", help="Subset of data (e.g., 'all', 'unguided', etc.)")
    parser.add_argument("--data_split", type=str, required=True, help="Data split (e.g., 'train', 'validation', 'test')")

    args = parser.parse_args()

    dataset = load_dataset(CHATBOT_ARENA_DATASET)
    generate_vpl_data(dataset=dataset, 
                            dir_name=args.output_dir, 
                            model_type=args.model_type, 
                            context_sample_strategy=args.context_sample_strategy,
                            num_random_contexts=args.num_random_contexts,
                            with_embeddings=args.with_embeddings, 
                            embedding_pool_strategy=args.embedding_pool_strategy,
                            data_subset=args.data_subset, 
                            data_split=args.data_split)

if __name__ == "__main__":
    main()