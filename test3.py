import torch

fake_qb = torch.tensor(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
#除以100
fake_qb = torch.div(fake_qb, 100)
values,indices = torch.topk(fake_qb, k=10)

print(values)
print(indices)

import os

print(os.path.dirname('../ad/asdsas'))
