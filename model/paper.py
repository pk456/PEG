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


# todo:检查cuda问题
class QB(object):
    def __init__(self, num_exers, num_concepts, q_to_k_file, exer_file):
        super().__init__()

        self.num_exers = num_exers
        self.num_concepts = num_concepts

        self.question_knowledge_mapping = {}
        with open(q_to_k_file, 'r') as f:
            for q in f:
                q = q.strip()  # 去除每行末尾的换行符 \n
                e, k = q.split('\t')  # 分割每行数据，使用制表符 \t 作为分隔符
                e = int(e)
                k = int(k) - self.num_exers  # k∈[0,122]
                if e in self.question_knowledge_mapping:  # 如果键已经存在，将值添加到对应的列表中
                    self.question_knowledge_mapping[e].append(k)
                else:
                    self.question_knowledge_mapping[e] = [k]

        self.exer_ids = []
        with open(exer_file, 'r') as f:
            for line in f:
                q = line.strip()  # 去除每行末尾的换行符 \n
                self.exer_ids.append(int(q))

    # 获取试题的知识点覆盖
    def get_knowledge_cover(self, paper):
        knowledge_counts = [0] * self.num_concepts
        for exer_id in paper:
            for knowledge_id in self.question_knowledge_mapping[exer_id]:
                knowledge_counts[knowledge_id] += 1

        sum_k = sum(knowledge_counts)
        cover = [count / sum_k for count in knowledge_counts]
        return cover, knowledge_counts

    def get_qb_cover(self):
        return self.get_knowledge_cover(self.question_knowledge_mapping)

    # 获取换题后的知识点覆盖率
    # todo:有时间看看，knowledge_counts应该直接就改变了，不需要再返回一次了,暂时没有用到
    def get_knowledge_cover_after_change(self, knowledge_counts, change_info):
        '''
        :param knowledge_counts:
        :param change_info:example{"remove": [1,2,3], "add": [4,5,6]}
        :return:
        '''
        knowledge_counts = copy.deepcopy(knowledge_counts)
        for exer_id in change_info["remove"]:
            for knowledge_id in self.question_knowledge_mapping[exer_id]:
                knowledge_counts[knowledge_id] -= 1
        for exer_id in change_info["add"]:
            for knowledge_id in self.question_knowledge_mapping[exer_id]:
                knowledge_counts[knowledge_id] += 1

        sum_k = sum(knowledge_counts)
        cover = [count / sum_k for count in knowledge_counts]
        return cover, knowledge_counts

    # 随机出题
    def generate_paper(self, num_q):
        return np.array(random.sample(self.question_knowledge_mapping.keys(), num_q))

    # 换题
    # todo:换题的时候可以换之前的题目吗？add的random.sample的第一个参数需要修改
    # todo:好像换题只能按顺序替换，那么就不能用set了
    def change_paper(self, paper, num_q, index):
        paper = copy.deepcopy(paper)
        remove = paper[index]
        # 至少卷子内的题不应该被抽取
        questions_to_choose = [q for q in self.question_knowledge_mapping.keys() if q not in paper]
        add = np.array(random.sample(questions_to_choose, num_q))
        paper[index] = add
        return paper, {"index": index, "remove": remove, "add": add}

    # 获取试卷中每道题目和知识点的关系
    def get_question_knowledges(self, paper):
        paper_knowledges = torch.zeros(len(paper), self.num_concepts)
        for index, exer_id in enumerate(paper):
            for knowledge_id in self.question_knowledge_mapping[exer_id]:
                paper_knowledges[index][knowledge_id] = 1
        return paper_knowledges

    def change_question_knowledges(self, paper_knowledges, change_info):
        paper_knowledges = copy.deepcopy(paper_knowledges)
        indices = change_info["index"]
        removes = change_info["remove"]
        adds = change_info["add"]
        for index, remove, add in zip(indices, removes, adds):
            for knowledge_id in self.question_knowledge_mapping[remove]:
                paper_knowledges[index][knowledge_id] = 0
            for knowledge_id in self.question_knowledge_mapping[add]:
                paper_knowledges[index][knowledge_id] = 1
        return paper_knowledges
