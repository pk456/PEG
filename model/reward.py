import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm, entropy
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pingouin as pg


# 知识点覆盖率
def skill_distance(qb_cover, paper_cover):
    qb_cover = np.array(qb_cover)
    # todo：paper_cover cuda
    paper_cover = np.array(paper_cover.detach().cpu().numpy())
    return np.linalg.norm(qb_cover - paper_cover)
    # return torch.norm(qb_cover - paper_cover)


def difficulty(avg_scores, avg):
    return np.abs(avg_scores - avg) / 100


# def div(R, Z):
#     log_Z = torch.log(Z)
#     # todo:这里为什么要log_Z，然后反过来
#     kl_div_result = F.kl_div(log_Z, R, reduction='sum')
#     return kl_div_result


def divergence(scores, truncated_normal, show=False):
    # 截断点
    lower_bound = 0  # 下限
    upper_bound = 100  # 上限
    # 计算经验分布的直方图
    num_bins = 10  # 可以根据需要调整bin的数量
    hist, bins = np.histogram(scores, bins=num_bins, density=True,
                              range=(lower_bound, upper_bound))  # density=True 使直方图归一化
    pk = (bins[1:] - bins[:-1]) * hist
    # 使用已知分布的pdf计算对应的概率
    bin_edges = np.linspace(lower_bound, upper_bound, num_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 计算每个bin的中心点
    known_probs = truncated_normal.pdf(bin_mids)
    qk = known_probs * np.diff(bin_edges).mean()
    # KL散度计算需要概率向量，因此需要确保它们之和接近1
    assert np.isclose(pk.sum(), 1, atol=1e-2), "Empirical distribution must sum to approximately 1"
    assert np.isclose(qk.sum(), 1,
                      atol=1e-2), "Known distribution probabilities must sum to approximately 1"
    # 计算KL散度
    kl_divergence = entropy(pk, qk, base=2)
    if show:
        # 绘制直方图
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=bins, alpha=0.7, density=True, color='blue', edgecolor='black')
        plt.plot(bin_mids, known_probs, 'r-', lw=5, alpha=0.6, label='truncnorm pdf')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.title('Truncated Normal Distribution')
        plt.legend()
        plt.grid(True)
        plt.show()
    return kl_divergence
    # return 1 / (1 + np.exp(-kl_divergence))


def divergence_1(scores):
    # 正态分布r2
    X = stats.truncnorm((0 - 70) / 15, (100 - 70) / 15, loc=70, scale=15)
    paper_distribution = X.rvs(100, random_state=23)  # seed在main.py第120行
    res = stats.wasserstein_distance(paper_distribution, scores) / len(scores)
    return res


def optimization_factor(scores, truncated_normal, qb_cover, paper_cover):
    # 知识点覆盖率
    dis = skill_distance(qb_cover, paper_cover)

    # 难度
    avg = truncated_normal.mean()
    avg_scores = np.mean(scores)
    dif = difficulty(avg_scores, avg)

    # 分布
    div = divergence(scores, truncated_normal)
    # div = divergence_1(scores)

    return np.mean([dis, dif, div]), 1 - dis, avg_scores / 100, 1 - div


class Reward(object):
    def __init__(self, truncated_normal, qb_cover):
        self.truncated_normal = truncated_normal
        self.qb_cover = qb_cover

    def optimization_factor(self, paper, student_concept_status, show_div=False):
        scores = paper.get_scores(student_concept_status).detach().cpu().numpy()
        paper_cover = paper.get_paper_cover()

        # 难度
        avg = self.truncated_normal.mean()
        avg_scores = np.mean(scores)
        dif = difficulty(avg_scores, avg)

        # 分布
        div = divergence(scores, self.truncated_normal, show_div)

        # 知识点覆盖率
        dis = skill_distance(self.qb_cover, paper_cover)
        return np.mean([dis, dif, div]), (avg_scores / 100, 1 - dis, 1 - div)


def evaluate_model(paper, student_concept_status, qb_cover):
    scores_list = paper.get_scores(student_concept_status).detach().cpu().numpy()
    p_list = paper.questions
    # 难度r1
    r1 = 1 - abs(sum(scores_list) / len(scores_list) - 70) / len(p_list)

    # 正态分布r2
    X = stats.truncnorm((0 - 70) / 15, (100 - 70) / 15, loc=70, scale=15)
    paper_distribution = X.rvs(100, random_state=23)
    r2 = 1 - stats.wasserstein_distance(paper_distribution, scores_list) / len(p_list)

    # 区分度r3
    scores_list.sort()  # len=50
    l1 = int(0.27 * len(scores_list))
    s1_dh = scores_list[:l1]
    s1_dl = scores_list[-l1:]
    a1 = sum(s1_dh) / len(s1_dh)
    a2 = sum(s1_dl) / len(s1_dl)
    r3 = 2.0 * (a2 - a1) / 100

    # 知识点覆盖率r4
    paper_cover = paper.get_paper_cover().detach().cpu().numpy()  # 试卷覆盖
    data_cover = qb_cover  # 题库覆盖
    r4 = cosine_similarity([data_cover], [paper_cover])[0][0]  # 知识点覆盖率：因为结果是二维列表，所以[0][0]取数值

    # 信度r5
    minmax_scaler = MinMaxScaler()
    data_normalized = minmax_scaler.fit_transform(
        paper.get_q_scores(student_concept_status).detach().cpu().numpy())  # 归一化
    data_normalized = pd.DataFrame(data_normalized)
    r5 = pg.cronbach_alpha(data_normalized)[0]

    r_5 = np.mean([r1, r2, r3, r4, r5])
    r_3 = np.mean([r1, r2, r4])
    return [r_5, r_3, r1, r2, r3, r4, r5]


if __name__ == '__main__':
    # 生成样本
    # 截断点
    lower_bound = 0  # 下限
    upper_bound = 100  # 上限
    # 将边界转换为标准化的z-score
    a = (lower_bound - 70) / 15
    b = (upper_bound - 70) / 15
    # 创建截断正态分布对象
    truncated_normal = truncnorm(a=a, b=b, loc=70, scale=15)
    truncated_normal2 = truncnorm(a=a, b=b, loc=70, scale=15)

    # 生成样本
    scores = truncated_normal.rvs(4000)
    div = divergence(scores, truncated_normal2, True)
    print(div)
