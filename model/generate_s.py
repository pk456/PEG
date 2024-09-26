import json
import os
import random

import torch
from data_preprocess.prepare_dataset import parse_student_seq, encode_onehot
from model.DKT import DKT

NUM_QUESTIONS = 122
BATCH_SIZE = 64
HIDDEN_SIZE = 10
NUM_LAYERS = 1

dkt = DKT(NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS)

dkt.load('../dkt.params')
dkt.dkt_model.eval()
with open('../data/c/log_data.json', 'r') as f:
    log_data = json.load(f)

all_seqs = []
for student in log_data:
    student_seq = parse_student_seq(student)
    student_data = encode_onehot([student_seq], student['log_num'], NUM_QUESTIONS)
    all_seqs.extend([student_data])

students = []

for student in all_seqs:
    student = torch.FloatTensor(student)
    s_k = dkt.dkt_model(student)
    students.append(s_k[-1][-1])
print(students)


def get_cover(input_dict):
    knowledge_counts = [0] * NUM_QUESTIONS
    # 遍历字典，统计每个知识点的累计考察次数
    for key, value in input_dict.items():
        for kwlg in value:
            knowledge_counts[kwlg] += 1

    sum_k = sum(knowledge_counts)  # 计算所有知识点的总考察次数
    cover = [count / sum_k for count in knowledge_counts]
    # print("sum_k", sum_k)
    # print("cover", cover)
    return cover


def get_all_dir():
    data_dict = {}
    data_file = f'../data/c/graph/e_to_k.txt'
    with open(data_file, 'r') as file:
        for line in file:
            line = line.strip()  # 去除每行末尾的换行符 \n
            e, k = line.split('\t')  # 分割每行数据，使用制表符 \t 作为分隔符
            e = int(e)
            k = int(k) - 17751  # k∈[0,122]
            if e in data_dict:  # 如果键已经存在，将值添加到对应的列表中
                data_dict[e].append(k)
            else:
                data_dict[e] = [k]
    # print("查看题目对应知识点", data_dict)  # len = 17671
    return data_dict


def get_paper_dict(pa_list, dict):
    p_dict = {}
    for e in pa_list:
        # e = e.item()  # 将张量元素转换为 Python 整数
        if e in dict:
            p_dict[e] = dict[e]

    return p_dict


QB = []  # 题库
with open(os.path.join("../data/c/", "exer.txt"), 'r') as file:
    for line in file:
        q = int(line.strip())
        QB.append(q)
print("QB", QB)
# 从QB中抽100道题
p_list = random.sample(QB, 100)

data_dict = get_all_dir()
paper_dict = get_paper_dict(p_list, data_dict)

# 计算知识点覆盖率
paper_cover = get_cover(paper_dict)
data_cover = get_cover(data_dict)


def e_k(paper_dict):
    result = []
    for key, value in paper_dict.items():
        knowledges = [0] * NUM_QUESTIONS
        for k in value:
            knowledges[k] = 1
        result.append(knowledges)
    return result


paper_ek = e_k(paper_dict)
paper_ek = torch.tensor(paper_ek)

def get_scores(students, paper_ek):
    scores = []
    for student in students:
        score = 0
        for r in paper_ek:
            score += torch.sum(r * student)
        scores.append(score)
    return scores

student_scores = get_scores(students, paper_ek)

from utils import dis, dif, div

distance = dis(paper_cover, data_cover)
difficulty = dif()

print()
