import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# 定义截断正态分布的参数
a, b = -2, 2  # 截断点
mu, sigma = 0, 1  # 均值和标准差

# 创建一个截断正态分布对象
tnorm = truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

# 生成符合该分布的数据集
data = tnorm.rvs(size=1000)

# 创建一个范围内的bin中心点
bin_edges = np.linspace(a, b, 50)
bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

# 计算每个bin_mid处的概率密度
pdf_values = tnorm.pdf(bin_mids)

# 绘制柱状图
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(data, bins=bin_edges, density=True, alpha=0.6, color='blue', edgecolor='black')

# 在同一图上绘制PDF曲线
plt.plot(bin_mids, pdf_values, 'r-', lw=2, label='truncnorm pdf')

# 设置图表标题和标签
plt.title('Histogram of Truncated Normal Distribution with PDF')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.grid(True)
plt.show()
