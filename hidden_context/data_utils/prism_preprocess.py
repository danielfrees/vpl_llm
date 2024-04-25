import argparse
import json
import os
import os.path as osp
from collections import defaultdict

from datasets import Dataset

import sys, ipdb, traceback


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()


sys.excepthook = info


def sort_by_user(data):
    data_sorted_by_users = defaultdict(list)
    for d in data:
        data_sorted_by_users[d['user_id']].append(d)
    users = sorted(data_sorted_by_users.keys())
    for user in users:
        if len(data_sorted_by_users[user]) != 6:
            data_sorted_by_users.pop(user)
    return data_sorted_by_users


def visualize_data(user_data):
    for c in user_data:
        print('\n\n', c['conversation_type'])
        for u in c['conversation_history']:
            if u['role'] == 'user':
                print("\n\nTurn {}\nUser: {}".format(u['turn'], u['content']))
            else:
                print("\nModel {} ({}): {}".format(u['within_turn_id'], u['model_name'], u['content']))
                print('Score: {}, {}'.format(u['score'], u['if_chosen']))
        print('\n\nPerformance attributes: {}'.format(c['performance_attributes']))
        print('Choice attributes: {}'.format(c['choice_attributes']))
        print('Open feedback: {}'.format(c['open_feedback']))


def group_user_data(data_sorted_by_users):
    data_grouped_by_users = defaultdict(list)
    for user in data_sorted_by_users.keys():
        c_type = {
            'unguided': [],
            'controversy guided': [],
            'values guided': []
        }
        for c in data_sorted_by_users[user]:
            c_type[c['conversation_type']].append(c)
        if len(c_type['unguided']) != 2 or len(c_type['controversy guided']) != 2 or len(c_type['values guided']) != 2:
            continue
    return data_grouped_by_users


def preprocess_data(data):
    data_sorted_by_users = sort_by_user(data)
    visualize_data(data_sorted_by_users['user1'])
    data_grouped_by_users = group_user_data(data_sorted_by_users)
    train_users, test_users = split_users(list(data_sorted_by_users.keys()))
    train_data = {data_sorted_by_users[user] for user in train_users}
    test_data = {data_sorted_by_users[user] for user in test_users}
    return train_data, test_data


def split_users(users):
    train_users = users[:int(len(users) * 0.9)]
    test_users = users[int(len(users) * 0.9):]
    return train_users, test_users


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/prism')
    parser.add_argument('--output_dir', type=str, default='data/prism')
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    conversations = list()
    with open(osp.join(data_dir, 'conversations.jsonl'), 'r') as f:
        for line in f:
            conversations.append(json.loads(line))

    train_dataset, test_dataset = preprocess_data(conversations)
    # os.makedirs(output_dir, exist_ok=True)
    # train_dataset.to_json(osp.join(output_dir, 'train.jsonl'))
    # test_dataset.to_json(osp.join(output_dir, 'test.jsonl'))
