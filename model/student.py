import json

import numpy as np
import torch

from data_preprocess.prepare_dataset import parse_student_seq
from model.DKT import DKT

NUM_QUESTIONS = 122
BATCH_SIZE = 64
HIDDEN_SIZE = 10
NUM_LAYERS = 1


# def convert_logs_to_one_hot_sequences(dataset, num_concepts):
#     with open(f'./data/{dataset}/log_data.json', 'r') as f:
#         log_data = json.load(f)
#     all_seqs = []
#     for student in log_data:
#         student_seq = parse_student_seq(student)
#
#         length = len(student_seq[0])
#         one_hot = np.zeros(shape=[length, num_concepts * 2])
#         i = 0
#         for log in student['logs']:
#             a = log['score']
#             for c_id in log['knowledge_code']:
#                 index = int(c_id + num_concepts if a > 0 else c_id)
#                 one_hot[i, index] = 1
#                 i += 1
#         all_seqs.append(one_hot)
#     return all_seqs

def convert_logs_to_one_hot_sequences(dataset, num_concepts):
    with open(f'./data/{dataset}/log_data.json', 'r') as f:
        log_data = json.load(f)
    all_seqs = []
    for student in log_data:
        # student_seq = parse_student_seq(student)

        length = student['log_num']
        one_hot = np.zeros(shape=[length, num_concepts * 2])
        i = 0
        for log in student['logs']:
            a = log['score']
            for c_id in log['knowledge_code']:
                index = int(c_id + num_concepts if a > 0 else c_id)
                one_hot[i, index] = 1
            i += 1
        all_seqs.append(one_hot)
    return all_seqs


def fetch_students_knowledge_status(students, model, num_concepts):
    students_knowledge_status = []
    for student in students:
        student = torch.FloatTensor(student.ravel())
        student = student.reshape([1, -1, 2 * num_concepts]).to(model.device)
        student_knowledge_status = model.dkt_model(student)[-1][-1]
        students_knowledge_status.append(student_knowledge_status)
    return torch.stack(students_knowledge_status)


if __name__ == '__main__':
    students = convert_logs_to_one_hot_sequences('c', 122)

    dkt = DKT(NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS, device=torch.device('cuda:0'))
    dkt.load('../dkt.params')
    dkt.dkt_model.eval()

    students_knowledge_status = fetch_students_knowledge_status(students, dkt, 122)
    print()
