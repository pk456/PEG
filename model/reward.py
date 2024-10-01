import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm, entropy
from scipy import stats


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


def divergence(scores, truncated_normal):
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

    def optimization_factor(self, paper, student_concept_status):
        scores = paper.get_scores(student_concept_status).detach().cpu().numpy()
        paper_cover = paper.get_paper_cover()

        # 难度
        avg = self.truncated_normal.mean()
        avg_scores = np.mean(scores)
        dif = difficulty(avg_scores, avg)

        # 分布
        div = divergence(scores, self.truncated_normal)

        # 知识点覆盖率
        dis = skill_distance(self.qb_cover, paper_cover)
        return np.mean([dis, dif, div]), (avg_scores / 100, 1 - dis, 1 - div)

if __name__ == '__main__':
    # 生成样本
    # 截断点
    lower_bound = 0  # 下限
    upper_bound = 100  # 上限
    # 将边界转换为标准化的z-score
    a = (lower_bound - 70) / 15
    b = (upper_bound - 70) / 15
    # 创建截断正态分布对象
    truncated_normal = truncnorm(a=a, b=b, loc=60, scale=15)
    truncated_normal2 = truncnorm(a=a, b=b, loc=70, scale=15)

    # 生成样本
    scores = truncated_normal.rvs(4000)
    div = divergence(scores, truncated_normal2)
    print(div)
