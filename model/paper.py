'''
    和试卷有关
    利用e_to_k.txt来建立试题和知识点的关系
    利用exer.txt存储了所有试题的编号，利用这个来抽题组卷
    这两个属性都是后续一直要用到的，所以建立一个class
'''
import copy
import random

import numpy as np
import torch







class Paper(object):
    def __init__(self, questions, paper_concepts):
        self.questions = questions
        self.paper_concepts = paper_concepts

    # 获取试题的知识点覆盖
    def get_paper_cover(self):
        # 获取试卷中每个知识点出现的次数
        paper_concept_counts = torch.sum(self.paper_concepts, dim=0)
        paper_concept_cover = torch.div(paper_concept_counts, torch.sum(paper_concept_counts))
        return paper_concept_cover

    def get_scores(self, students_concept_status):
        '''
            :param paper_concepts:shape [num_exer, num_concepts]
            :param students_concept_status: shape [num_students, num_concepts]
            :return:
            '''
        students_concept_status = students_concept_status.unsqueeze(1).expand(-1, self.paper_concepts.shape[0], -1)
        concept_match = students_concept_status * self.paper_concepts
        mask = torch.ne(concept_match, 0)
        concept_match = torch.where(mask, concept_match, 1)
        students_q_score = torch.prod(concept_match, dim=-1)
        return torch.sum(students_q_score, dim=-1)
