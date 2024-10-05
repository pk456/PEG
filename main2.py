import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import truncnorm
from torch.utils.data import DataLoader

from generate_paper.pdp_eg import PDP_EG
from generate_paper.pga_eg import PGA_EG
from model.DKT import DKT
from model.ExamGAN import ExamGAN, ExamDataset
from model.paper import Paper
from model.qb import QB
from model.reward import Reward, optimization_factor, evaluate_model
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
    parser.add_argument('--method', type=str, default='gan', help='pdp or pga or gan')

    parser.add_argument('--random_seed', type=int, default=23)  # 用于计算分布stats.wasserstein_distance而增加
    parser.add_argument('--epoch', type=int, default=50, help='number of update paper')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    # 数据集相关设置
    parser.add_argument('--dataset', type=str, default='c_filter2', help='dataset')
    parser.add_argument('--all_num_questions', type=int, default=15161, help='number of questions in qb')
    parser.add_argument('--num_concepts', type=int, default=116, help='number of concept')
    parser.add_argument('--train_data_file', type=str, default='gan/train_data.pkl',
                        help='train data file,拼接dataset路径，只针对gan网络')

    # gan网络设置
    parser.add_argument('--batch_size', type=int, default=32, help='batch size,，只针对gan网络')
    parser.add_argument('--random_dim', type=int, default=100, help='random_dim,，只针对gan网络')
    parser.add_argument('--lr', type=float, default=0.001, help='random_dim,，只针对gan网络')
    parser.add_argument('--gan_save_path', type=str, default='./saved_models/c_filter2/exam_gan', help='只针对gan网络')

    # DKT网络设置
    parser.add_argument('--load_model', type=str, default='./saved_models4/model19', help='load model')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers.')

    # 班级试题相关设置
    parser.add_argument('--num_students', type=int, default=2627, help='number of students')
    # 试题相关设置
    parser.add_argument('--num_questions', type=int, default=100, help='number of questions each paper')
    parser.add_argument('--num_init_papers', type=int, default=1000, help='number of initial papers')
    parser.add_argument('--mean', type=float, default=70, help='')
    parser.add_argument('--std', type=float, default=15, help='')
    parser.add_argument('--save_paper', type=bool, default=False, help='pdp or pga or gan')
    parser.add_argument('--save_paper_path', type=str, default='./papers/', help='pdp or pga or gan')


    # 遗传相关设置
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='crossover rate')
    parser.add_argument('--mutation_rate', type=float, default=0.003, help='mutation rate')
    return parser.parse_args()


def run(peg, args, students_concept_status):
    logging.info('init...')
    paper = peg.init(n=args.num_init_papers, num_q=args.num_questions)
    logging.info('update...')
    paper = peg.update(paper, args.num_questions, args.epoch)
    logging.info('done')
    optimized_factor = reward.optimization_factor(paper, students_concept_status)
    logging.info(f"optimized_factor:{optimized_factor}")
    return paper


def run2(exam_gan, args, students_concept_status, qb):
    logging.info('init...')
    if os.path.exists(args.gan_save_path + '_generator') and os.path.exists(args.gan_save_path + '_discriminator'):
        exam_gan.load_model(args.gan_save_path)
    else:
        train_dataset = ExamDataset(os.path.join('data', args.dataset, args.train_data_file))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        exam_gan.train(train_data=train_loader, batch_size=args.batch_size, epoch=args.epoch, lr=args.lr)
        if not os.path.exists(os.path.dirname(args.gan_save_path)):
            os.makedirs(os.path.dirname(args.gan_save_path))
        exam_gan.save_model(args.gan_save_path)
    student_concepts = students_concept_status.T.detach()
    c = torch.cat((torch.mean(student_concepts, -1), torch.std(student_concepts, -1)))
    fake_qb = exam_gan.generate_exam_script(c)
    _, questions = torch.topk(fake_qb, k=args.num_questions)
    questions = questions.detach().cpu().numpy()
    return Paper(questions, qb.get_question_concepts(questions))


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

    if args.method == 'pdp':
        peg = PDP_EG(qb, students_concept_status, reward)
    elif args.method == 'pga':
        peg = PGA_EG(qb, students_concept_status, reward, args)
    elif args.method == 'gan':
        peg = ExamGAN(args)
    else:
        raise ValueError('method must be pdp or pga')

    evaluation = []
    for i in range(20):
        if args.method == 'gan':
            paper = run2(peg, args, students_concept_status, qb)
        else:
            paper = run(peg, args, students_concept_status)
        # 保存
        if args.save_paper is not None:
            if os.path.exists(os.path.join(args.save_paper_path, args.dataset, args.method)) is False:
                os.makedirs(os.path.join(args.save_paper_path, args.dataset, args.method))
            paper.save(os.path.join(args.save_paper_path, args.dataset, args.method, f'paper{i}.pkl'))

        # 评估
        result = evaluate_model(paper, students_concept_status, qb_cover=qb.get_qb_cover())
        evaluation.append(result)
    evaluation = np.array(evaluation)
    logging.info(f'{args.method} evaluation: mean--{evaluation.mean(0)}, std--{evaluation.std(0)}')
    # 保存
    if args.save_paper is not None:
        with open(os.path.join(args.save_paper_path, args.dataset, args.method, 'evaluation.csv'), 'wb') as f:
            df = pd.DataFrame(evaluation)
            df.to_csv(f, index=False,
                      header=['四维度均值', '三维度均值', '难度r1', '正态分布r2', '区分度r3', '知识点覆盖率r4',
                              '信度r5'])
