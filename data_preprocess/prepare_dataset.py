import json
import random

import numpy as np
import tqdm


def parse_student_seq(student):
    q_all = []
    a_all = []
    for log in student['logs']:
        a = log['score']
        for q in log['knowledge_code']:
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
        for i, q_id in enumerate(q):
            index = int(q_id + num_concepts if a[i] > 0 else q_id)
            onehot[i][index] = 1
        result = np.append(result, onehot)

    return result.reshape(-1, max_step, 2 * num_concepts)


def train_test_split(data, train_size=.7, shuffle=True):
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_size)
    return data[: boundary], data[boundary:]


def preprocess(args):
    with open(f'./data/{args.dataset}/log_data.json', 'r') as f:
        log_data = json.load(f)
    all_seqs = []
    for student in log_data:
        student_seq = parse_student_seq(student)
        all_seqs.extend([student_seq])

    train_sequences, test_sequences = train_test_split(all_seqs, args.train_size, args.shuffle)

    train_data = encode_onehot(train_sequences, args.max_len, args.num_concepts)
    test_data = encode_onehot(test_sequences, args.max_len, args.num_concepts)

    np.save(f'./data/{args.dataset}/train_data.npy', train_data)
    np.save(f'./data/{args.dataset}/test_data.npy', test_data)
