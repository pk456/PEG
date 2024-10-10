import os

from torch.utils.data import DataLoader

from model.ExamGAN import ExamDataset
from model.paper import Paper,load
import numpy as np
import torch

paper1 = load(r'F:\ProgramData\kt\re\PEG\papers\assistment\gan\paper0.pkl')
paper2 = load(r'F:\ProgramData\kt\re\PEG\papers\assistment\gan\paper1.pkl')
paper3 = load(r'F:\ProgramData\kt\re\PEG\papers\assistment\gan\paper46.pkl')
q = set(paper3.questions) & set(paper2.questions)
print(q)
print(len(q))
train_dataset = ExamDataset(os.path.join('data', 'assistment', 'gan/train_data.pkl'))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
for c, qb, students_concept_status in train_loader:
    print(qb)
print(paper1)
print(paper2)


'''
2:98
4:98
7:99
'''
