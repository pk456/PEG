import argparse
import logging
import random

import torch
from scipy.stats import truncnorm

from generate_paper.pdp_eg import PDP_EG
from generate_paper.pga_eg import PGA_EG
from model.DKT import DKT
from model.qb import QB
from model.reward import Reward
from model.student import convert_logs_to_one_hot_sequences, fetch_students_concept_status

logging.getLogger().setLevel(logging.INFO)


def init_sq(args):
    # 读取学生知识掌握状态
    students = convert_logs_to_one_hot_sequences(args.dataset, args.num_concepts)
    students = random.sample(students, args.num_students)
    dkt = DKT(args.num_concepts, args.hidden_size, args.num_layers, device=torch.device(f'cuda:{args.gpu}'))
    dkt.load(args.load_model)
    dkt.dkt_model.eval()
    students_concept_status = fetch_students_concept_status(students, dkt, args.num_concepts)

    # 读取试卷信息
    qb = QB(args.all_num_questions, args.num_concepts, f'./data/{args.dataset}/graph/e_to_k.txt',
            f'./data/{args.dataset}/exer.txt')

    return qb, students_concept_status


def ctl():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=23)  # 用于计算分布stats.wasserstein_distance而增加
    parser.add_argument('--epoch', type=int, default=50, help='number of update paper')
    parser.add_argument('--num_init_papers', type=int, default=1000, help='number of initial papers')
    parser.add_argument('--num_questions', type=int, default=100, help='number of questions each paper')
    parser.add_argument('--num_students', type=int, default=4163, help='number of students')
    parser.add_argument('--num_concepts', type=int, default=122, help='number of concept')
    parser.add_argument('--dataset', type=str, default='c', help='dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--load_model', type=str, default='./saved_models/models19', help='load model')
    parser.add_argument('--all_num_questions', type=int, default=17751, help='number of questions in qb')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers.')
    parser.add_argument('--mean', type=float, default=60, help='')
    parser.add_argument('--std', type=float, default=15, help='')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='crossover rate')
    parser.add_argument('--mutation_rate', type=float, default=0.003, help='mutation rate')
    return parser.parse_args()


if __name__ == '__main__':
    args = ctl()

    # 生成样本
    # 截断点
    lower_bound = 0  # 下限
    upper_bound = 100  # 上限
    # 将边界转换为标准化的z-score
    a = (lower_bound - args.mean) / args.std
    b = (upper_bound - args.mean) / args.std
    # 创建截断正态分布对象
    truncated_normal = truncnorm(a=a, b=b, loc=args.mean, scale=args.std)
    qb, students_concept_status = init_sq(args)
    reward = Reward(truncated_normal, qb_cover=qb.get_qb_cover())

    peg = PDP_EG(qb, students_concept_status, reward)
    # peg = PGA_EG(qb, students_concept_status, reward, args)
    logging.info('init...')
    paper = peg.init(n=args.num_init_papers, num_q=args.num_questions)
    logging.info('update...')
    paper = peg.update(paper, args.num_questions, args.epoch)
    logging.info('done')
