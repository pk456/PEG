'''
给定一个班级的学生，通过执行以下五个步骤来创建训练数据：
（i）基于练习记录，捕获学生在班级中的知识掌握水平（即c），作为生成考试试卷的条件（详见第4.1节）。
（ii）在所有知识点中，我们根据它们在考试题库中的频率随机挑选n个知识点，即频率越高的知识点越有可能被选中。
（iii）对于这n个知识点中的每一个，它可能被题库中的多个问题覆盖，从中随机选择一个问题并插入到考试试卷中。这个过程重复执行，
    直到考试试卷包含n个问题。通过这种方式，创建了m个考试试卷（本研究中m=1000）。
（iv）对于每个新创建的m个考试试卷，使用知识追踪来估计学生得分，这需要学生完成的练习记录，如第3.1节中解释的。获得估计学生得分的分布RS_Ei；
    并计算REi和Z之间的相似度，按照公式(11)计算。
（v）在所有m个考试试卷中，选择与Z相似度最高的1%。对于每个被采用的考试试卷，创建一个训练数据实例<c;VE>，其中VE是一个向量，
    每个元素对应考试题库中的一个问题。VE中对应考试试卷中的n个问题的元素是1；其他元素的值是0。
'''
import argparse
import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.stats import truncnorm

from model.DKT import DKT
from model.paper import Paper
from model.qb import QB
from model.reward import Reward, divergence
from model.student import fetch_students_concept_status, convert_logs_to_one_hot_sequences

'''
需要有一个知识点在题库中的频率，方便第二步抽知识点
需要有一个知识点和习题的对应关系,方便第三步取抽题
'''
logging.getLogger().setLevel(logging.INFO)


class QB_GAN(QB):
    def __init__(self, num_exers, num_concepts, q_to_k_file, exer_file):
        super().__init__(num_exers, num_concepts, q_to_k_file, exer_file)
        self.concept_question_mapping = {}
        with open(q_to_k_file, 'r') as f:
            for q in f:
                q = q.strip()  # 去除每行末尾的换行符 \n
                e, k = q.split('\t')  # 分割每行数据，使用制表符 \t 作为分隔符
                e = int(e)
                k = int(k) - self.num_exers  # k∈[0,122]
                if k in self.concept_question_mapping:  # 如果键已经存在，将值添加到对应的列表中
                    self.concept_question_mapping[k].append(e)
                else:
                    self.concept_question_mapping[k] = [e]

    def select_concepts_by_frequency(self, n):
        qb_cover = self.get_qb_cover()
        return random.choices(range(self.num_concepts), weights=qb_cover, k=n)

    def generate_paper_by_concepts(self, concepts):
        questions = []
        for concept in concepts:
            question = random.choice(self.concept_question_mapping[concept])
            questions.append(question)
        paper_concepts = self.get_question_concepts(questions)
        return Paper(questions, paper_concepts)


def init_sq_gan(args):
    # 读取学生知识掌握状态
    students = convert_logs_to_one_hot_sequences(args.dataset, args.num_concepts)
    students = random.sample(students, args.num_students)
    dkt = DKT(args.num_concepts, args.hidden_size, args.num_layers, device=torch.device(f'cuda:{args.gpu}'))
    dkt.load(args.load_model)
    dkt.dkt_model.eval()
    with torch.no_grad():
        students_concept_status = fetch_students_concept_status(students, dkt, args.num_concepts)

    # 读取试卷信息
    qb = QB_GAN(args.all_num_questions, args.num_concepts, f'./data/{args.dataset}/graph/e_to_k.txt',
                f'./data/{args.dataset}/exer.txt')

    return qb, students_concept_status


def ctl():
    parser = argparse.ArgumentParser()
    # 数据集相关设置
    parser.add_argument('--dataset', type=str, default='c_filter2', help='dataset')
    parser.add_argument('--all_num_questions', type=int, default=15161, help='number of questions in qb')
    parser.add_argument('--num_concepts', type=int, default=116, help='number of concept')
    # DKT网络相关设置
    parser.add_argument('--load_model', type=str, default='./saved_models/c_filter2/model19', help='load model')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers.')
    # 班级试题相关设置
    parser.add_argument('--num_students', type=int, default=50, help='number of students in a class')
    parser.add_argument('--num_class', type=int, default=100, help='')
    # 试题相关设置
    parser.add_argument('--num_init_papers', type=int, default=1000, help='number of initial papers')
    parser.add_argument('--num_questions', type=int, default=100, help='number of questions each paper')
    parser.add_argument('--mean', type=float, default=70, help='')
    parser.add_argument('--std', type=float, default=15, help='')
    # 其他设置
    parser.add_argument('--gpu', type=int, default=0, help='gpu')  #
    parser.add_argument('--train_rate', type=float, default=0.8, help='')  #
    parser.add_argument('--val_rate', type=float, default=0.1, help='')  #
    return parser.parse_args()


def create_class(args):
    # 生成样本
    # 截断点
    lower_bound = 0  # 下限
    upper_bound = 100  # 上限
    # 将边界转换为标准化的z-score
    a = (lower_bound - args.mean) / args.std
    b = (upper_bound - args.mean) / args.std
    # 创建截断正态分布对象
    truncated_normal = truncnorm(a=a, b=b, loc=args.mean, scale=args.std)
    qb, students_concept_status = init_sq_gan(args)

    logging.info('creating train data...')
    result = []
    for i in tqdm.tqdm(range(args.num_init_papers)):
        concepts = qb.select_concepts_by_frequency(args.num_questions)
        paper = qb.generate_paper_by_concepts(concepts)
        scores = paper.get_scores2(students_concept_status)
        z = 1 - divergence(scores.detach().cpu().numpy(), truncated_normal, show=False)
        result.append({'z': z, 'paper': paper})

    # 降序排列
    result.sort(key=lambda x: x['z'], reverse=True)
    papers = result[:int(len(result) * 0.01)]
    # 试卷挑选完毕
    logging.info('papers selected')

    # 开始整理数据
    logging.info('preparing data...')
    # 创建训练数据
    data = []
    student_concepts = students_concept_status.T.detach().cpu().numpy()
    c = [student_concepts.mean(-1), student_concepts.std(-1)]
    for item in papers:
        paper = item['paper']
        ve = np.zeros(args.all_num_questions)
        for q in paper.questions:
            ve[q] = 1
        data.append((c, ve, students_concept_status.detach().cpu().numpy()))
    logging.info('finished')
    return data


def create_data(args, type):
    data = []
    if type == 'train':
        rate = args.train_rate
    elif type == 'val':
        rate = args.val_rate
    else:
        rate = 1 - args.train_rate - args.val_rate
    for i in range(int(args.num_class * rate)):
        data.extend(create_class(args))
    if not os.path.exists(f'data/{args.dataset}/gan'):
        os.makedirs(f'data/{args.dataset}/gan')
    save(data, f'data/{args.dataset}/gan/{type}_data.pkl')


def save(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    args = ctl()
    create_data(args, 'train')
    create_data(args, 'val')
    create_data(args, 'test')
