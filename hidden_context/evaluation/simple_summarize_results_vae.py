# This file is revised: only evaluate baselines trained on "both" dataset
import argparse
import glob

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm

import sys, ipdb, traceback


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()


sys.excepthook = info


if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, help="path to directory containing eval_results_both.jsonl"
    )
    parser.add_argument(
        "--ckpt_name", type=str, help="checkpoint directory name", default=None
    )
    args = parser.parse_args()

    print("\n========== DETECTING HIDDEN CONTEXT RESULTS ==========")
    for reward_model_type in ["vae"]:
        if args.ckpt_name is None:
            ckpt_dir = f"{reward_model_type}_*_peft_last_checkpoint"
        else:
            ckpt_dir = args.ckpt_name
        print(f"\n--- Reward model type: {reward_model_type} ---")
        for train_set in ["both"]:
            checkpoint_dir = glob.glob(
                f"{args.dir}/" + ckpt_dir
            )[0]
            hh_rlhf_evaluation = pd.read_json(
                f"{checkpoint_dir}/eval_results_both.jsonl", lines=True
            )

            chosen_reward_outputs = np.array(
                hh_rlhf_evaluation.reward_output_chosen.tolist()
            )
            rejected_reward_outputs = np.array(
                hh_rlhf_evaluation.reward_output_rejected.tolist()
            )


            def explained_variance(mean, stdev):
                var_in_means = np.var(mean)
                mean_var = np.mean(stdev ** 2)
                return var_in_means / (var_in_means + mean_var)


            def get_reward_mean_and_stdev(reward_outputs):
                if isinstance(reward_model_type, list):
                    reward_outputs = np.array(reward_outputs)
                    return np.mean(reward_outputs, axis=1), np.std(reward_outputs, axis=1)
                return reward_outputs, np.ones_like(reward_outputs)

            chosen_mean, chosen_stdev = get_reward_mean_and_stdev(chosen_reward_outputs)
            rejected_mean, rejected_stdev = get_reward_mean_and_stdev(
                rejected_reward_outputs
            )
            r2 = explained_variance(
                np.concatenate([chosen_mean, rejected_mean]),
                np.concatenate([chosen_stdev, rejected_stdev]),
            )

            print(f"Model trained on {train_set} dataset(s): r² = {r2}")

            uncontroversial_evaluation = hh_rlhf_evaluation[
                hh_rlhf_evaluation.controversial == False
            ]
            controversial_evaluation = hh_rlhf_evaluation[
                hh_rlhf_evaluation.controversial == True
            ]

            uncontroversial_chosen_reward_outputs = np.array(
                uncontroversial_evaluation.reward_output_chosen.tolist()
            )
            uncontroversial_rejected_reward_outputs = np.array(
                uncontroversial_evaluation.reward_output_rejected.tolist()
            )

            controversial_chosen_reward_outputs = np.array(
                controversial_evaluation.reward_output_chosen.tolist()
            )
            controversial_rejected_reward_outputs = np.array(
                controversial_evaluation.reward_output_rejected.tolist()
            )

            def get_mean_reward(reward_outputs):
                return np.mean(reward_outputs, axis=1)

            print(
                f"Accuracy on simple uncontroversial data for model trained on {train_set} dataset(s):",
                np.mean(
                    get_mean_reward(uncontroversial_chosen_reward_outputs)
                    >= get_mean_reward(uncontroversial_rejected_reward_outputs)
                ),
            )
            print(
                f"Accuracy on simple controversial data for model trained on {train_set} dataset(s):",
                np.mean(
                    get_mean_reward(controversial_chosen_reward_outputs)
                    >= get_mean_reward(controversial_rejected_reward_outputs)
                ),
            )
