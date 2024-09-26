'''
    和试卷有关
    利用e_to_k.txt来建立试题和知识点的关系
    利用exer.txt存储了所有试题的编号，利用这个来抽题组卷
    这两个属性都是后续一直要用到的，所以建立一个class
'''
import random

import torch


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
    # todo:有时间看看，knowledge_counts应该直接就改变了，不需要再返回一次了
    def get_knowledge_cover_after_change(self, knowledge_counts, change):
        '''
        :param knowledge_counts:
        :param change:example{"remove": [1,2,3], "add": [4,5,6]}
        :return:
        '''
        for exer_id in change["remove"]:
            for knowledge_id in self.question_knowledge_mapping[exer_id]:
                knowledge_counts[knowledge_id] -= 1
        for exer_id in change["add"]:
            for knowledge_id in self.question_knowledge_mapping[exer_id]:
                knowledge_counts[knowledge_id] += 1

        sum_k = sum(knowledge_counts)
        cover = [count / sum_k for count in knowledge_counts]
        return cover, knowledge_counts

    # 随机出题
    def generate_paper(self, num_q):
        return set(random.sample(self.question_knowledge_mapping.keys(), num_q))

    # 换题
    # todo:换题的时候可以换之前的题目吗？add的random.sample的第一个参数需要修改
    def change_paper(self, paper, num_q):
        remove = set(random.sample(paper, num_q))
        add = set(random.sample(self.question_knowledge_mapping.keys(), num_q))
        paper = paper - remove | add
        return paper, {"remove": remove, "add": add}

    # 获取试卷中每道题目和知识点的关系
    # todo: 换题的时候，需要重新计算，看后续如何优化。有一个思路是直接整个保存一个这样的对应向量，这里直接组合就行。
    def get_question_knowledges(self, paper):
        paper_knowledges = torch.zeros(len(paper), self.num_concepts)
        for index, exer_id in paper:
            for knowledge_id in self.question_knowledge_mapping[exer_id]:
                paper_knowledges[index][knowledge_id] = 1
        return paper_knowledges
