import json
import random

import numpy as np
import tqdm


def parse_student_seq(student):
    q_all = []
    a_all = []
    for log in student['logs']:
        a = log['score']
        q = []
        for k in log['knowledge_code']:
            q.append(k)
        q_all.append(q)
        a_all.append(a)
    return q_all, a_all


def encode_onehot(sequences, max_step, num_concepts):
    result = []

    for q, a in tqdm.tqdm(sequences, 'convert to one-hot format: '):
        length = len(q)
        # append questions' and answers' length to an integer multiple of max_step
        mod = 0 if length % max_step == 0 else (max_step - length % max_step)
        onehot = np.zeros(shape=[length + mod, 2 * num_concepts])
        for i, c_ids in enumerate(q):
            for c_id in c_ids:
                index = int(c_id + num_concepts if a[i] > 0 else c_id)
                onehot[i][index] = 1
        result = np.append(result, onehot)

    return result.reshape(-1, max_step, 2 * num_concepts)


def train_test_split(data, train_size=.7, shuffle=True):
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_size)
    return data[: boundary], data[boundary:]


def preprocess(args):
    dataset = args['dataset']
    with open(f'../data/{dataset}/log_data.json', 'r') as f:
        log_data = json.load(f)
    all_seqs = []
    for student in log_data:
        student_seq = parse_student_seq(student)
        all_seqs.extend([student_seq])

    train_sequences, test_sequences = train_test_split(all_seqs, args['train_size'], args['shuffle'])

    train_data = encode_onehot(train_sequences, args['max_len'], args['num_concepts'])
    test_data = encode_onehot(test_sequences, args['max_len'], args['num_concepts'])

    np.save(f'../data/{dataset}/train_data2.npy', train_data)
    np.save(f'../data/{dataset}/test_data2.npy', test_data)


if __name__=='__main__':
    args = {
        "dataset": 'c',
        "train_size": 0.7,
        "shuffle": True,
        "max_len": 1254,
        "num_concepts": 122
    }
    preprocess(args)
