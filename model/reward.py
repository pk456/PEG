import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm, entropy


# 知识点覆盖率
def skill_distance(qb_cover, paper_cover):
    qb_cover = np.array(qb_cover)
    paper_cover = np.array(paper_cover)
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
    hist, bins = np.histogram(scores, bins=num_bins, density=True)  # density=True 使直方图归一化
    pk = (bins[1:] - bins[:-1]) * hist
    # 使用已知分布的pdf计算对应的概率
    bin_edges = np.linspace(lower_bound, upper_bound, num_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 计算每个bin的中心点
    known_probs = truncated_normal.pdf(bin_mids)
    qk = known_probs * np.diff(bin_edges).mean()
    # KL散度计算需要概率向量，因此需要确保它们之和接近1
    assert np.isclose(pk.sum(), 1, atol=1e-2), "Empirical distribution must sum to approximately 1"
    assert np.isclose(known_probs.sum() * np.diff(bin_edges).mean(), 1,
                      atol=1e-2), "Known distribution probabilities must sum to approximately 1"
    # 计算KL散度
    kl_divergence = entropy(pk, qk, base=2)

    return kl_divergence


def optimization_factor(scores, truncated_normal, qb_cover, paper_cover):
    # 知识点覆盖率
    dis = skill_distance(qb_cover, paper_cover)

    # 难度
    avg = truncated_normal.mean()
    avg_scores = np.mean(scores)
    dif = difficulty(avg_scores, avg)

    # 分布
    div = divergence(scores, truncated_normal)

    return np.mean([dis, dif, div]), 1-dis, avg_scores/100, 1-div



