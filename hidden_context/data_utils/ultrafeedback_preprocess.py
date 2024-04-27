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


def get_user_type(chosen_ratings, rejected_ratings):
    keys = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    chosen_rating_values = list()
    rejected_chosen_values = list()
    for key in keys:
        chosen_rating_values.append(chosen_ratings[key])
        rejected_chosen_values.append(rejected_ratings[key])
    chosen_values = np.asarray(chosen_rating_values)
    rejected_values = np.asarray(rejected_chosen_values)
    return keys[random_argmax(chosen_values - rejected_values)]


def inner_join(original, binarized):
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
        # if bin_idx == 16837:
        #     ipdb.set_trace()
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
        data_subset = get_user_type(chosen_ratings, rejected_ratings)
        dataset_dict['Index'].append(out_idx)
        dataset_dict['prompt'].append(prompt)
        dataset_dict['chosen'].append('Human: ' + prompt + '\n\nAssistant: ' + chosen)
        dataset_dict['rejected'].append('Human: ' + prompt + '\n\nAssistant: ' + rejected)
        dataset_dict['data_subset'].append(data_subset)
        out_idx += 1
    return Dataset.from_dict(dataset_dict)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    ultra_feedback = load_dataset('openbmb/UltraFeedback')
    binarized_cleaned = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned')
    print(len(binarized_cleaned['train']))
    joined_dataset = inner_join(ultra_feedback['train'], binarized_cleaned['train'])

    output_dir = os.path.join('data', 'UltraFeedback')
    for aspect in ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']:
        subset = joined_dataset.filter(lambda x: x['data_subset'] == aspect)
        print(aspect, len(subset))
        split = subset.train_test_split(test_size=0.1)
        train_split = split['train']
        test_split = split['test']
        train_split.to_json(os.path.join(output_dir, aspect, 'train.jsonl'))
        test_split.to_json(os.path.join(output_dir, aspect, 'test.jsonl'))





