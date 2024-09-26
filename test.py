import numpy as np
from scipy.stats import truncnorm, entropy

# 正态分布的参数
mu = 50  # 均值
sigma = 15  # 标准差

# 截断点
lower_bound = 0  # 下限
upper_bound = 100  # 上限

# 将边界转换为标准化的z-score
a = (lower_bound - mu) / sigma
b = (upper_bound - mu) / sigma

# 创建截断正态分布对象
truncated_normal = truncnorm(a=a, b=b, loc=mu, scale=sigma)


# 生成未知分布的数据集
unknown_data = np.random.uniform(low=0, high=100, size=1000)  # 生成1000个0到100的随机数
# 计算经验分布的直方图
num_bins = 100  # 可以根据需要调整bin的数量
hist, bins = np.histogram(unknown_data, bins=num_bins, density=True)  # density=True 使直方图归一化

# 使用已知分布的pdf计算对应的概率
bin_mids = bins[:-1] + np.diff(bins) / 2  # 计算每个bin的中心点
known_probs = truncated_normal.pdf(bin_mids)

# KL散度计算需要概率向量，因此需要确保它们之和接近1
assert np.isclose(hist.sum(), 1, atol=1e-2), "Empirical distribution must sum to approximately 1"
assert np.isclose(known_probs.sum() * np.diff(bins).mean(), 1, atol=1e-2), "Known distribution probabilities must sum to approximately 1"

# 计算KL散度 D(P || Q)，其中P是已知分布，Q是经验分布
# 注意：scipy的entropy函数计算的是逆方向的KL散度，即D(Q || P)
kl_divergence = entropy(hist, known_probs)

print("KL Divergence:", kl_divergence)
