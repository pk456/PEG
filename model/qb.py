import copy
import random
import numpy as np
import torch

from model.paper import Paper


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
    def _get_paper_cover(self, questions):
        knowledge_counts = [0] * self.num_concepts
        for exer_id in questions:
            for knowledge_id in self.question_knowledge_mapping[exer_id]:
                knowledge_counts[knowledge_id] += 1

        sum_k = sum(knowledge_counts)
        cover = [count / sum_k for count in knowledge_counts]
        return cover, knowledge_counts

    def get_qb_cover(self):
        qb_cover, _ = self._get_paper_cover(self.question_knowledge_mapping)
        return qb_cover

    # 随机出题
    def generate_paper(self, num_q):
        questions = np.array(random.sample(list(self.question_knowledge_mapping.keys()), num_q))
        # 获取试卷对应的知识点
        paper_concepts = self.get_question_concepts(questions)

        return Paper(questions, paper_concepts)

    # 换题
    def change_paper(self, paper: Paper, index):
        new_paper = copy.deepcopy(paper)
        add = np.array(random.sample(list(self.question_knowledge_mapping.keys()), len(index)))
        remove = new_paper.questions[index]
        new_paper.questions[index] = add

        change_info = {"index": index, "remove": remove, "add": add}
        new_paper.paper_concepts = self._change_question_concepts(new_paper.paper_concepts, change_info)

        return new_paper, change_info

    def exchange_paper(self, paper1, paper2, pt):
        # todo:之后改下，重复太多了
        new_paper1 = copy.deepcopy(paper1)
        new_paper2 = copy.deepcopy(paper2)
        add1 = new_paper2.questions[:pt]
        add2 = new_paper1.questions[:pt]
        remove1 = new_paper1.questions[:pt]
        remove2 = new_paper2.questions[:pt]
        new_paper1.questions[:pt] = add1
        new_paper2.questions[:pt] = add2
        change_info1 = {"index": list(range(pt)), "remove": remove1, "add": add1}
        change_info2 = {"index": list(range(pt)), "remove": remove2, "add": add2}
        new_paper1.paper_concepts = self._change_question_concepts(new_paper1.paper_concepts, change_info1)
        new_paper2.paper_concepts = self._change_question_concepts(new_paper2.paper_concepts, change_info2)
        return [new_paper1, new_paper2]

    # 获取试卷中每道题目和知识点的关系
    def get_question_concepts(self, questions):
        # todo：paper cuda问题
        paper_concepts = torch.zeros(len(questions), self.num_concepts).to(torch.device('cuda:0'))
        index = 0
        for exer_id in questions:
            for knowledge_id in self.question_knowledge_mapping[exer_id]:
                paper_concepts[index][knowledge_id] = 1
            index += 1
        return paper_concepts

    def _change_question_concepts(self, paper_concepts, change_info):
        paper_concepts = copy.deepcopy(paper_concepts)
        indices = change_info["index"]
        removes = change_info["remove"]
        adds = change_info["add"]
        for index, remove, add in zip(indices, removes, adds):
            for knowledge_id in self.question_knowledge_mapping[remove]:
                paper_concepts[index][knowledge_id] = 0
            for knowledge_id in self.question_knowledge_mapping[add]:
                paper_concepts[index][knowledge_id] = 1
        return paper_concepts


if __name__ == "__main__":
    qb = QB(17751, 122, f'../data/c/graph/e_to_k.txt',
            f'../data/c/exer.txt')
    paper = qb.generate_paper(100)

    paper2 = qb.change_paper(paper, [0])
