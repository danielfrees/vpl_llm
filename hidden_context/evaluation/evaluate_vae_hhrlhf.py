# This file is used to evaluate VAE model on HH-RLHF dataset
# TODO: Not compatible with new framework yet!!!
import os
from dataclasses import dataclass, field
from typing import Optional, cast

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from hidden_context.train_llm_preference_model import (
    DataSubset,
    get_hh_rlhf_dataset,
)

from hidden_context.train_llm_vae_preference_model import HHRLHFPreprocessor, RewardDataCollatorWithPadding

from hidden_context.vae_utils import VAEModel, Annealer
import numpy as np


import sys, ipdb, traceback


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()


sys.excepthook = info


@dataclass
class ScriptArguments:
    reward_model_checkpoint: str = field(
        metadata={"help": "Path to the trained reward model checkpoint."}
    )
    checkpoint_name: str = field(
        metadata={"help": "Directory name of the trained reward model checkpoint."}
    )
    output: Optional[str] = field(
        default=None,
        metadata={"help": "JSONL file where results will be stored."},
    )
    batch_size: Optional[int] = field(default=1)
    model_name: Optional[str] = field(default="gpt2")
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
            "for your model",
        },
    )
    data_path: str = field(
        default="Anthropic/hh-rlhf",
    )
    data_subset: str = field(
        default="both",
        metadata={
            "help": "Which subset of the data to use. You can choose between 'both', "
            "'helpful', or 'harmless'."
        },
    )
    eval_dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    num_outputs: int = field(
        default=1024,
        metadata={"help": "The number of outputs from the model."},
    )
    max_length: int = field(default=1024)
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 precision."},
    )
    latent_dim: int = field(default=512)
    hidden_dim: int = field(default=512)
    embed_dim: int = field(default=1024)
    fixed_contexts: bool = field(
        default=True,
        metadata={"help": "Whether to use pre-calculated fixed contexts embeddings."}
    )
    controversial_only: bool = field(default=False)
    fixed_llm_embeddings: bool = field(default=False)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    data_subset = cast(DataSubset, script_args.data_subset)
    checkpoint_path = os.path.join(script_args.reward_model_checkpoint, script_args.checkpoint_name)
    output_fname = script_args.output
    if output_fname is None:
        output_fname = os.path.join(
            checkpoint_path, f"eval_results_{data_subset}.jsonl"
        )

    eval_dataset = get_hh_rlhf_dataset(
        data_subset,
        "test",
        script_args.eval_dataset_size,
        data_path=script_args.data_path,
    )
    if script_args.controversial_only:
        eval_dataset = eval_dataset.filter(lambda example: example['controversial'] == True)

    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    # peft_config = LoraConfig.from_pretrained(checkpoint_path)
    # model_kwargs = {}
    # if script_args.bf16:
    #     model_kwargs["torch_dtype"] = torch.bfloat16
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     script_args.model_name,
    #     num_labels=script_args.num_outputs,
    #     **model_kwargs,
    # )
    # model = PeftModel.from_pretrained(
    #     model, checkpoint_path, is_trainable=False
    # )

    # Need to do this for GPT2 and Llama because they don't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    num_proc = 24  # Can adjust to be higher if you have more processors.
    num_samples = script_args.num_outputs
    eval_dataset = eval_dataset.map(
        HHRLHFPreprocessor(script_args, tokenizer),
        batched=True,
        num_proc=num_proc,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["max_lengths"] <= script_args.max_length
    )
    embed_dim = script_args.embed_dim
    latent_dim = script_args.latent_dim
    hidden_dim = script_args.hidden_dim
    model = torch.load(os.path.join(
        script_args.reward_model_checkpoint,
        script_args.checkpoint_name,
        'model.pt'
    ))

    collator = RewardDataCollatorWithPadding(args=script_args,
                                             tokenizer=tokenizer,
                                             padding=True,
                                             max_length=script_args.max_length,
                                             pad_to_multiple_of=64)

    def compute_rewards(example):
        device = model.llm_encoder.device
        outputs = {}
        inputs = collator([example])
        batch_size = inputs["seq_start_end"]
        if model.fixed_llm_embeddings:
            embeddings_chosen = torch.tensor(inputs["embeddings_chosen"]).to(device).bfloat16()
            embeddings_rejected = torch.tensor(inputs["embeddings_rejected"]).to(device).bfloat16()
        else:
            embeddings = model.llm_encoder(
                torch.concatenate(
                    [
                        inputs["input_ids_chosen"],
                        inputs["input_ids_rejected"],
                    ],
                    dim=0,
                ),
                torch.concatenate(
                    [
                        inputs["attention_mask_chosen"],
                        inputs["attention_mask_rejected"],
                    ],
                    dim=0,
                ),
            )[0]
            embeddings_chosen = embeddings[:batch_size]
            embeddings_rejected = embeddings[batch_size:]

        if model.fixed_contexts:
            contexts_embeddings_chosen = torch.tensor(inputs["contexts_embeddings_chosen"]).to(device).bfloat16()
            contexts_embeddings_rejected = torch.tensor(inputs["contexts_embeddings_rejected"]).to(device).bfloat16()
        else:
            contexts_embeddings_chosen = model.llm_encoder(
                inputs["contexts_input_ids_chosen"],
                inputs["contexts_attention_mask_chosen"]
            )[0]
            contexts_embeddings_rejected = model.llm_encoder(
                inputs["contexts_input_ids_rejected"],
                inputs["contexts_attention_mask_rejected"]
            )[0]
        seq_start_end = inputs["seq_start_end"]

        rewards_chosen, rewards_rejected, mean, log_var = model(
            embeddings_chosen,
            embeddings_rejected,
            contexts_embeddings_chosen,
            contexts_embeddings_rejected,
            seq_start_end,
        )

        outputs[f"reward_output_chosen"] = [rewards_chosen.mean().item()]
        outputs[f"reward_output_rejected"] = [rewards_rejected.mean().item()]
        outputs["latent_mean"] = mean.tolist()
        outputs["latent_log_var"] = log_var.tolist()
        return outputs

    eval_results = eval_dataset.map(
        compute_rewards,
        batched=False,
    )

    eval_results.to_json(output_fname, orient="records", lines=True)

