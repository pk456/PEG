import json
import random

import numpy as np
import tqdm


def parse_student_seq2(student):
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


def encode_onehot2(sequences, max_step, num_concepts):
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


