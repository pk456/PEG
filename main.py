import argparse
import random

import numpy as np
import torch
import tqdm
from scipy.stats import truncnorm

from model.DKT2 import DKT
from model.qb import QB
from model.reward import optimization_factor
from model.student import convert_logs_to_one_hot_sequences, fetch_students_concept_status


def calculate_exam_score(paper_concepts, students_concept_status):
    '''
    :param paper_concepts:shape [num_exer, num_concepts]
    :param students_concept_status: shape [num_students, num_concepts]
    :return:
    '''
    students_concept_status = students_concept_status.unsqueeze(1).expand(-1, paper_concepts.shape[0], -1)
    concept_match = students_concept_status * paper_concepts
    mask = torch.ne(concept_match, 0)
    concept_match = torch.where(mask, concept_match, 1)
    students_q_score = torch.prod(concept_match, dim=-1)
    return torch.sum(students_q_score, dim=-1)


def generate_clipped_normal_samples(mean, variance, n, low=0, upper=100, *, seed=0):
    """
    生成符合正态分布的样本，并确保所有样本值都在区间 [0, 100] 内。

    参数:
    mean (float): 正态分布的均值。
    variance (float): 正态分布的方差。
    n (int): 需要生成的样本数量。

    返回:
    numpy.ndarray: 包含 n 个样本的 NumPy 数组，所有样本值都在 [0, 100] 区间内。
    """
    # 设置随机种子以保证结果可复现
    np.random.seed(seed)

    std_dev = np.sqrt(variance)  # 方差的平方根即为标准差
    valid_samples = []

    while len(valid_samples) < n:
        candidate_sample = np.random.normal(loc=mean, scale=std_dev)
        if low <= candidate_sample <= upper:
            valid_samples.append(candidate_sample)

    return np.array(valid_samples)


def init(args):
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


def init_paper(qb, num_question, truncated_normal, students_concept_status):
    paper = qb.generate_paper(num_question)
    paper_concepts = qb.get_question_concepts(paper).to(students_concept_status.device)
    # 计算学生成绩
    scores = calculate_exam_score(paper_concepts, students_concept_status)
    scores = scores.detach().to(torch.device('cpu')).numpy()

    qb_cover, _ = qb.get_qb_cover()
    paper_cover, concept_counts = qb._get_paper_cover(paper)
    optimized_factor, dis, dif, div = optimization_factor(scores, truncated_normal, qb_cover, paper_cover)
    return {
        'paper': paper,
        'paper_concepts': paper_concepts,
        'scores': scores,
        'paper_cover': paper_cover,
        'concept_counts': concept_counts,
        'optimized_factor': optimized_factor,
        'dis': dis,
        'dif': dif,
        'div': div
    }


def update(qb, paper_concepts, truncated_normal, students_concept_status, concept_counts, change_info):
    paper_concepts = qb.change_question_concepts(paper_concepts, change_info).to(
        (students_concept_status.device))
    # 计算学生成绩
    scores = calculate_exam_score(paper_concepts, students_concept_status)
    scores = scores.detach().to(torch.device('cpu')).numpy()

    qb_cover, _ = qb.get_qb_cover()
    paper_cover, concept_counts = qb.get_concept_cover_after_change(concept_counts, change_info)
    optimized_factor, dis, dif, div = optimization_factor(scores, truncated_normal, qb_cover, paper_cover)
    return {
        'paper_concepts': paper_concepts,
        'scores': scores,
        'qb_cover': qb_cover,
        'paper_cover': paper_cover,
        'concept_counts': concept_counts,
        'optimized_factor': optimized_factor,
        'dis': dis,
        'dif': dif,
        'div': div
    }


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
    parser.add_argument('--load_model', type=str, default='./saved_models/model39', help='load model')
    parser.add_argument('--all_num_questions', type=int, default=17751, help='number of questions in qb')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers.')
    parser.add_argument('--mean', type=float, default=70, help='')
    parser.add_argument('--std', type=float, default=15, help='')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='crossover rate')
    parser.add_argument('--mutation_rate', type=float, default=0.003, help='mutation rate')
    return parser.parse_args()


def pdp_eg(args):
    # 生成样本
    # 截断点
    lower_bound = 0  # 下限
    upper_bound = 100  # 上限
    # 将边界转换为标准化的z-score
    a = (lower_bound - args.mean) / args.std
    b = (upper_bound - args.mean) / args.std
    # 创建截断正态分布对象
    truncated_normal = truncnorm(a=a, b=b, loc=args.mean, scale=args.std)
    # 初始化
    qb, students_concept_status = init(args)

    min_paper_info = None
    for i in range(args.num_init_papers):
        paper_info = init_paper(qb, args.num_questions, truncated_normal, students_concept_status)
        if min_paper_info is None or paper_info['optimized_factor'] < min_paper_info['optimized_factor']:
            min_paper_info = paper_info

    paper = min_paper_info['paper']
    # 迭代
    for i in tqdm.tqdm(range(args.epoch)):
        if i >= args.num_questions:
            break
        new_paper, change_info = qb.change_paper(paper, [i])
        new_paper_info = update(qb, min_paper_info['paper_concepts'], truncated_normal, students_concept_status,
                                min_paper_info['concept_counts'], change_info)
        if new_paper_info['optimized_factor'] < min_paper_info['optimized_factor']:
            min_paper_info = new_paper_info
            min_paper_info['paper'] = new_paper

    return min_paper_info


def pga_eg(args):
    # 生成样本
    # 截断点
    lower_bound = 0  # 下限
    upper_bound = 100  # 上限
    # 将边界转换为标准化的z-score
    a = (lower_bound - args.mean) / args.std
    b = (upper_bound - args.mean) / args.std
    # 创建截断正态分布对象
    truncated_normal = truncnorm(a=a, b=b, loc=args.mean, scale=args.std)
    # 初始化
    qb, students_concept_status = init(args)

    # 初始群落
    paper_pop = []
    for i in range(args.num_init_papers):
        paper_info = init_paper(qb, args.num_questions, truncated_normal, students_concept_status)
        paper_pop.append(paper_info)
    # todo: 因为适应度高的更容易被选中，加上是放回抽样，所以很有可能两次选的是一个个体
    for i in tqdm.tqdm(range(args.epoch)):
        print('第{}次迭代'.format(i))
        new_paper_pop = []
        while len(new_paper_pop) < args.num_init_papers:
            # 选择父代
            offsprings = selection(paper_pop,
                                   [1 - paper['optimized_factor'] for paper in paper_pop],
                                   2)
            if random.random() <= args.crossover_rate:
                # 交叉
                offsprings = crossover(offsprings, args.num_questions, qb, truncated_normal, students_concept_status)
                if random.random() <= args.mutation_rate:
                    # 变异
                    offsprings = mutation(offsprings, args.num_questions, qb, truncated_normal,
                                          students_concept_status)
                new_paper_pop.extend(offsprings)
        paper_pop = new_paper_pop
    max_paper_info = None
    for final_paper in paper_pop:
        if max_paper_info is None or final_paper['optimized_factor'] > max_paper_info['optimized_factor']:
            max_paper_info = final_paper
    return max_paper_info


def selection(paper_pop, ads, num_parents):
    # todo: 适应度高的更容易被选中，加上是放回抽样，所以很有可能两次选的是一个个体
    parents = random.choices(paper_pop, ads, k=num_parents)
    return parents


def crossover(parents, num_questions, qb, truncated_normal, students_concept_status):
    pt = random.randint(1, num_questions - 2)
    paper1 = np.concatenate((parents[0]['paper'][:pt], parents[1]['paper'][pt:]))
    offspring = get_paper_info(qb, paper1, truncated_normal, students_concept_status)
    paper2 = np.concatenate((parents[1]['paper'][:pt], parents[0]['paper'][pt:]))
    offspring2 = get_paper_info(qb, paper2, truncated_normal, students_concept_status)
    return [offspring, offspring2]


def get_paper_info(qb, paper, truncated_normal, students_concept_status):
    paper_concepts = qb.get_question_concepts(paper).to(students_concept_status.device)
    # 计算学生成绩
    scores = calculate_exam_score(paper_concepts, students_concept_status)
    scores = scores.detach().to(torch.device('cpu')).numpy()

    qb_cover, _ = qb.get_qb_cover()
    paper_cover, concept_counts = qb._get_paper_cover(paper)
    optimized_factor, dis, dif, div = optimization_factor(scores, truncated_normal, qb_cover, paper_cover)
    return {
        'paper': paper,
        'paper_concepts': paper_concepts,
        'scores': scores,
        'paper_cover': paper_cover,
        'concept_counts': concept_counts,
        'optimized_factor': optimized_factor,
        'dis': dis,
        'dif': dif,
        'div': div
    }


def mutation(offsprings, num_questions, qb, truncated_normal, students_concept_status):
    mut_papers = []
    # todo: 目前先随机替换一个吧
    for offspring in offsprings:
        k = random.randint(0, num_questions - 1)
        new_paper, change_info = qb.change_paper(offspring['paper'], [k])
        new_paper_info = update(qb, offspring['paper_concepts'], truncated_normal, students_concept_status,
                                offspring['concept_counts'], change_info)
        new_paper_info['paper'] = new_paper
        mut_papers.append(new_paper_info)
    return mut_papers


if __name__ == '__main__':
    args = ctl()
    dif = []
    val = []
    div = []
    for i in range(100):
        print('第', i, '次: ', end='')
        paper_info = pdp_eg(args)
        dif.append(paper_info['dif'])
        val.append(paper_info['dis'])
        div.append(paper_info['div'])
    print("\t\tdif\t\tdis\t\tdiv")
    print('mean\t', np.mean(dif), '\t', np.mean(val), '\t', np.mean(div))
    print('std\t', np.std(dif), '\t', np.std(val), '\t', np.std(div))
