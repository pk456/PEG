import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

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

# 计算概率密度函数
x = np.linspace(lower_bound, upper_bound, 100)
y = truncated_normal.pdf(x)

# 绘制概率密度函数
plt.plot(x, y)
plt.title('Truncated Normal Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.show()

# 计算累积分布函数
cdf_values = truncated_normal.cdf(x)

# 绘制累积分布函数
plt.plot(x, cdf_values)
plt.title('Truncated Normal Distribution CDF')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.show()

# 生成随机样本
samples = truncated_normal.rvs(size=1000, random_state=42)

# 绘制随机样本的直方图
plt.hist(samples, bins=30, density=True)
plt.title('Random Samples from Truncated Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
