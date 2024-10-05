import random

import tqdm

from generate_paper.pdp_eg import PEG


class PGA_EG(PEG):
    def __init__(self, qb, student_concept_status, reward, args):
        super().__init__()
        self.qb = qb
        self.student_concept_status = student_concept_status
        self.reward = reward
        self.crossover_rate = args.crossover_rate
        self.mutation_rate = args.mutation_rate

    def init(self, n, num_q):
        paper_pop = []
        for i in range(n):
            paper_pop.append(self.qb.generate_paper(num_q))
        return paper_pop

    def update(self, init_data, num_q, epoch):
        paper_pop = init_data
        for i in tqdm.tqdm(range(epoch)):
            new_paper_pop = []
            optimization_factors = [
                1 - self.reward.optimization_factor(paper, self.student_concept_status)[0] for paper in paper_pop]
            while len(new_paper_pop) < len(init_data):
                offsprings = self._selection(paper_pop, optimization_factors, 2)
                if random.random() <= self.crossover_rate:
                    offsprings = self._crossover(offsprings, num_q)
                if random.random() <= self.mutation_rate:
                    offsprings = self._mutation(offsprings, num_q)
                new_paper_pop.extend(offsprings)
            paper_pop = new_paper_pop

        best_paper = None
        best_optimized_factor = None
        for paper in paper_pop:
            optimized_factor = 1 - self.reward.optimization_factor(paper, self.student_concept_status)[0]
            if (best_paper is None and best_optimized_factor is None) or optimized_factor > best_optimized_factor:
                best_paper = paper
                best_optimized_factor = optimized_factor
        return best_paper

    def _selection(self, paper_pop, optimization_factors, k):
        parents = random.choices(paper_pop, optimization_factors, k=k)
        return parents

    def _crossover(self, parents, num_q):
        pt = random.randint(1, num_q - 2)
        return self.qb.exchange_paper(parents[0], parents[1], pt)

    def _mutation(self, offsprings, num_q):
        mut_papers = []
        for paper in offsprings:
            k = random.randint(1, num_q - 1)
            new_paper, _ = self.qb.change_paper(paper, [k])
            mut_papers.append(new_paper)
        return mut_papers
