import time
import multiprocessing
import numpy as np
from sklearn.model_selection import ParameterGrid

import cartpole

N_RUNS = 10

# buckets = []
# for a in range(1,7):
#     for b in range(1,7):
#         for c in range(3,7):
#             for d in range(3,7):
#                 buckets.append((a,b,c,d,))

buckets = []
for a in range(1,4,2):
    for b in range(1,4,2):
        for c in range(5,7):
            for d in range(5,7):
                buckets.append((a,b,c,d,))

grid_params = {
    'min_alpha': [0.1],
    'buckets': buckets,
    'gamma': [0.99],
    'min_epsilon': [0.001],
    'max_discount': [0]
}

grid = list(ParameterGrid(grid_params))
final_score = np.zeros(len(grid))
threads = []

def _evaluate_params(args):
    index, params = args
    print('Evaluating params: {}'.format(params))

    scores = []
    for i in range(N_RUNS):
        agent = cartpole.QLearnerCartPole(**params)
        score = agent.run()
        scores.append(score)

    score = np.mean(scores)
    print('Finished evaluating set {} with score of {}.'.format(index, score))
    return score

def run():
    start_time = time.time()
    print('About to evaluate {} parameter sets.'.format(len(grid)))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    final_scores = pool.map(_evaluate_params, list(enumerate(grid)))

    print('Best parameter set was {} with score of {}'.format(grid[np.argmin(final_scores)], np.min(final_scores)))
    print('Worst parameter set was {} with score of {}'.format(grid[np.argmax(final_scores)], np.max(final_scores)))
    print('Execution time: {} sec'.format(time.time() - start_time))

if __name__ == '__main__':
    run()