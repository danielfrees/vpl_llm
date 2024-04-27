import argparse
import random

from datasets import load_dataset, Dataset
import numpy as np
import os

import sys, ipdb, traceback


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()


sys.excepthook = info


def random_argmax(values):
    """ a random tie-breaking argmax """
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


def random_greater_than_zero(values):
    return (np.random.randn(values.shape[0]) * (values == 0) > 0.0) | (values > 0.0)


def array_to_type(arr):
    return str(int(np.dot(arr, np.array([8, 4, 2, 1]))))


def get_user_type(chosen_ratings, rejected_ratings, augment_type, users):
    keys = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    chosen_rating_values = list()
    rejected_chosen_values = list()
    for key in keys:
        chosen_rating_values.append(chosen_ratings[key])
        rejected_chosen_values.append(rejected_ratings[key])
    chosen_values = np.asarray(chosen_rating_values)
    rejected_values = np.asarray(rejected_chosen_values)
    if augment_type == 'single':
        pass
    elif augment_type == 'set':
        pass
    elif augment_type == 'pos_neg':
        user_orig = np.ones(4, dtype=int) * (random_greater_than_zero(chosen_values - rejected_values))
        user_rev = 1 - user_orig
        data_subsets = [array_to_type(user_orig), array_to_type(user_rev)]
        reversed_labels = [False, True]
        return data_subsets, reversed_labels
    else:
        raise ValueError('Invalid augment_type')

    return keys[random_argmax(chosen_values - rejected_values)]


def inner_join(original, binarized, augment_type, users):
    keys = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    orig_idx = 0
    out_idx = 0
    dataset_dict = {
        'Index': list(),
        'prompt': list(),
        'chosen': list(),
        'rejected': list(),
        'data_subset': list(),
    }
    for bin_idx in range(len(binarized)):
        while binarized[bin_idx]['prompt'] != original[orig_idx]['instruction']:
            orig_idx += 1
        prompt = binarized[bin_idx]['prompt']
        chosen = binarized[bin_idx]['chosen'][1]['content']
        rejected = binarized[bin_idx]['rejected'][1]['content']
        if chosen == '' or rejected == '':
            continue
        chosen_ratings = dict()
        rejected_ratings = dict()
        flag = True
        for c in original[orig_idx]['completions']:
            if c['response'] == chosen:
                for key in keys:
                    r = c['annotations'][key]['Rating']
                    if r == 'N/A':
                        flag = False
                        continue
                    chosen_ratings[key] = int(r)
            elif c['response'] == rejected:
                for key in keys:
                    r = c['annotations'][key]['Rating']
                    if r == 'N/A':
                        flag = False
                        continue
                    rejected_ratings[key] = int(r)
            else:
                continue
        if not flag or len(chosen_ratings) != 4 or len(rejected_ratings) != 4:
            continue

        data_subsets, reversed_labels = get_user_type(chosen_ratings, rejected_ratings, augment_type, users)
        for idx, data_subset in enumerate(data_subsets):
            dataset_dict['Index'].append(out_idx)
            dataset_dict['prompt'].append(prompt)
            if not reversed_labels[idx]:
                dataset_dict['chosen'].append('Human: ' + prompt + '\n\nAssistant: ' + chosen)
                dataset_dict['rejected'].append('Human: ' + prompt + '\n\nAssistant: ' + rejected)
            else:
                dataset_dict['chosen'].append('Human: ' + prompt + '\n\nAssistant: ' + rejected)
                dataset_dict['rejected'].append('Human: ' + prompt + '\n\nAssistant: ' + chosen)
            dataset_dict['data_subset'].append(data_subset)
            out_idx += 1
    return Dataset.from_dict(dataset_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-a', '--augment_type', type=str, default=None, help='How to augment data')
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    sixteen = {
        '0': (0, 0, 0, 0),
        '1': (0, 0, 0, 1),
        '2': (0, 0, 1, 0),
        '3': (0, 0, 1, 1),
        '4': (0, 1, 0, 0),
        '5': (0, 1, 0, 1),
        '6': (0, 1, 1, 0),
        '7': (0, 1, 1, 1),
        '8': (1, 0, 0, 0),
        '9': (1, 0, 0, 1),
        '10': (1, 0, 1, 0),
        '11': (1, 0, 1, 1),
        '12': (1, 1, 0, 0),
        '13': (1, 1, 0, 1),
        '14': (1, 1, 1, 0),
        '15': (1, 1, 1, 1),
    }
    if args.augment_type == 'single':
        user_types = {
            '8': (1, 0, 0, 0),
            '4': (0, 1, 0, 0),
            '2': (0, 0, 1, 0),
            '1': (0, 0, 0, 1),
        }
    elif args.augment_type == 'set':
        user_types = sixteen.copy()
        user_types.pop('0')
    elif args.augment_type == 'pos_neg':
        user_types = sixteen
    else:
        raise ValueError('Invalid augment_type')

    ultra_feedback = load_dataset('openbmb/UltraFeedback')
    binarized_cleaned = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned')
    print(len(binarized_cleaned['train']))
    joined_dataset = inner_join(ultra_feedback['train'], binarized_cleaned['train'], args.augment_type, user_types)

    output_dir = os.path.join('data', 'UltraFeedback_{}'.format(args.augment_type))
    for user_type in user_types.keys():
        subset = joined_dataset.filter(lambda x: x['data_subset'] == user_type)
        print(user_types[user_type], len(subset))
        split = subset.train_test_split(test_size=0.1)
        train_split = split['train']
        test_split = split['test']
        train_split.to_json(os.path.join(output_dir, user_type, 'train.jsonl'))
        test_split.to_json(os.path.join(output_dir, user_type, 'test.jsonl'))

