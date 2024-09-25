import json
import random

import numpy as np
import tqdm

MAX_STEP = 200
NUM_QUESTIONS = 122


def parse_student_seq(student):
    q_all = []
    a_all = []
    for log in student['logs']:
        a = log['score']
        for q in log['knowledge_code']:
            q_all.append(q)
            a_all.append(a)
    return q_all, a_all


def encode_onehot(sequences, max_step, num_questions):
    result = []

    for q, a in tqdm.tqdm(sequences, 'convert to one-hot format: '):
        length = len(q)
        # append questions' and answers' length to an integer multiple of max_step
        mod = 0 if length % max_step == 0 else (max_step - length % max_step)
        onehot = np.zeros(shape=[length + mod, 2 * num_questions])
        for i, q_id in enumerate(q):
            index = int(q_id if a[i] > 0 else q_id + num_questions)
            onehot[i][index] = 1
        result = np.append(result, onehot)

    return result.reshape(-1, max_step, 2 * num_questions)


def train_test_split(data, train_size=.7, shuffle=True):
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_size)
    return data[: boundary], data[boundary:]


if __name__ == '__main__':
    with open('../data/c/log_data.json', 'r') as f:
        log_data = json.load(f)

    max_len = 200
    result = 0
    for student in log_data:
        if student['log_num'] > max_len:
            result += 1
    print(result)

    all_seqs = []
    for student in log_data:
        student_seq = parse_student_seq(student)
        all_seqs.extend([student_seq])

    train_sequences, test_sequences = train_test_split(all_seqs)

    train_data = encode_onehot(train_sequences, MAX_STEP, NUM_QUESTIONS)
    test_data = encode_onehot(test_sequences, MAX_STEP, NUM_QUESTIONS)

    np.save('../data/c/train_data.npy', train_data)
    np.save('../data/c/test_data.npy', test_data)
