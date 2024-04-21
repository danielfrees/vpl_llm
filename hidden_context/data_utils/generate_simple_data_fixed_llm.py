# This file is used to generate synthetic language dataset
import os
from dataclasses import dataclass, field
from typing import Optional, cast

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import torch

from hidden_context.data_utils.data_processing import (
    ScriptArguments,
    generate_embeddings_with_llm,
    generate_contexts
)

from hidden_context.data_utils.simple_templates import (
    helpful_harmless_sentences,
    helpful_harmful_sentences,
    harmless_unhelpful_sentences,
    harmful_unhelpful_sentences
)

from hidden_context.train_llm_preference_model import (
    DataSubset,
    get_hh_rlhf_dataset,
    concatenate_datasets,
    HHRLHFPreprocessor,
)

from copy import deepcopy

import numpy as np

import sys, ipdb, traceback


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()


sys.excepthook = info


def generate_synthetic_dataset_with_embeddings(args):
    data_subset = cast(DataSubset, args.data_subset)
    input_dataset = get_hh_rlhf_dataset(
        data_subset,
        args.data_split,
        args.dataset_size,
        data_path=args.data_path,
        use_subset_as_dir=True
    )
    dim = args.embed_dim // 2

    # helpful_harmless = torch.cat([torch.ones(dim), torch.ones(args.embed_dim - dim)])
    # helpful_harmful = torch.cat([torch.ones(dim), -torch.ones(args.embed_dim - dim)])
    # harmless_unhelpful = torch.cat([-torch.ones(dim), torch.ones(args.embed_dim - dim)])
    # harmful_unhelpful = torch.cat([-torch.ones(dim), -torch.ones(args.embed_dim - dim)])
    helpful_harmless = torch.cat([torch.ones(1), torch.ones(1), torch.zeros(args.embed_dim - 2)])
    helpful_harmful = torch.cat([torch.ones(1), -torch.ones(1), torch.zeros(args.embed_dim - 2)])
    harmless_unhelpful = torch.cat([-torch.ones(1), torch.ones(1), torch.zeros(args.embed_dim - 2)])
    harmful_unhelpful = torch.cat([-torch.ones(1), -torch.ones(1), torch.zeros(args.embed_dim - 2)])

    def sample_from_distribution(center, std=0.1):
        return torch.normal(center, std)

    def generate_simple_data_point(example):
        pair_type = np.random.randint(6)
        if pair_type == 0:
            chosen = sample_from_distribution(helpful_harmless)
            rejected = sample_from_distribution(helpful_harmful)
        elif pair_type == 1:
            chosen = sample_from_distribution(harmless_unhelpful)
            rejected = sample_from_distribution(harmful_unhelpful)
        elif pair_type == 2:
            chosen = sample_from_distribution(helpful_harmless)
            rejected = sample_from_distribution(harmless_unhelpful)
        elif pair_type == 3:
            chosen = sample_from_distribution(helpful_harmful)
            rejected = sample_from_distribution(harmful_unhelpful)
        elif pair_type == 4:
            chosen = sample_from_distribution(helpful_harmless)
            rejected = sample_from_distribution(harmful_unhelpful)
        else:
            if script_args.data_subset == 'helpful':
                chosen = sample_from_distribution(helpful_harmful)
                rejected = sample_from_distribution(harmless_unhelpful)
            else:
                chosen = sample_from_distribution(harmless_unhelpful)
                rejected = sample_from_distribution(helpful_harmful)
        return_dict = {'prompt': "", 'chosen': "", 'rejected': "", 'responses': ["", ""], 'embeddings': {
            'embedding_chosen': chosen,
            'embedding_rejected': rejected,
        }}
        if pair_type == 5:
            return_dict['controversial'] = True
        else:
            return_dict['controversial'] = False
        return return_dict

    input_dataset = input_dataset.map(generate_simple_data_point)

    return input_dataset


if __name__ == "__main__":
    # default setting on synthetic language dataset, please iterate over data subsets and data splits
    np.random.seed(0)
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    dataset = generate_synthetic_dataset_with_embeddings(script_args)
    generate_contexts(script_args, dataset)

# python -m hidden_context.data_utils.generate_simple_data_fixed_llm
# --output_dir data/simple_gaussian/
# --data_path data/relabeled_hh_rlhf --data_subset helpful --data_split test --dataset_size 100
# --add_controversial True --synthetic_dataset True
