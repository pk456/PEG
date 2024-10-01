import tqdm


class PEG(object):
    def __init__(self, qb, student_concept_status, reward):
        self.qb = qb
        self.student_concept_status = student_concept_status
        self.reward = reward

    def init(self, n, num_q):
        pass

    def update(self, init_data, num_q, epoch):
        pass


def better_paper(optimized_factor, best_optimized_factor):
    return optimized_factor < best_optimized_factor


class PDP_EG(PEG):
    def __init__(self, qb, student_concept_status, reward):
        super().__init__(qb, student_concept_status, reward)

    def init(self, n, num_q):
        best_paper = None
        best_optimized_factor = None
        for i in tqdm.tqdm(range(n)):
            paper = self.qb.generate_paper(num_q)
            optimized_factor, _ = self.reward.optimization_factor(paper, self.student_concept_status)

            if ((best_paper is None and best_optimized_factor is None) or
                    better_paper(optimized_factor, best_optimized_factor)):
                best_optimized_factor = optimized_factor
                best_paper = paper
        return best_paper, best_optimized_factor

    def update(self, init_data, num_q, epoch):
        best_paper = init_data
        best_optimized_factor, _ = self.reward.optimization_factor(init_data, self.student_concept_status)

        for i in tqdm.tqdm(range(epoch)):
            if i >= num_q:
                break
            new_paper, _ = self.qb.change_paper(init_data, [i])
            optimized_factor, _ = self.reward.optimization_factor(new_paper, self.student_concept_status)
            if better_paper(optimized_factor, best_optimized_factor):
                best_optimized_factor = optimized_factor
                best_paper = new_paper
        return best_paper, best_optimized_factor
