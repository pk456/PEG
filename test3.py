import random

import numpy as np

population = ['A', 'B', 'C']
weights = [0.2, 0.3, 0.5]

# 这将引发 ValueError
selected = random.choices(population, cum_weights=weights, k=2)
print(selected)

selected = np.random.choice(population, size=2, replace=False, p=weights)
print(selected)

print(population)
