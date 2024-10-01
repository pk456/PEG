import json
import numpy as np
import torch


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


def fetch_students_concept_status(students, model, num_concepts):
    students_concept_status = []
    for student in students:
        student = torch.FloatTensor(student.ravel())
        student = student.reshape([1, -1, 2 * num_concepts]).to(model.device)
        student_concept_status = model.dkt_model(student)[-1][-1]
        students_concept_status.append(student_concept_status)
    return torch.stack(students_concept_status)
