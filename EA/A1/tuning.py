from typing import List

import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import studentnumber1_studentnumber2_GA, GA_1, create_problem
from skopt import gp_minimize
budget = 1000000
# 
# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

# Hyperparameters to tune, e.g.
hyperparameter_space = {
    "population_size": [50, 100, 200],
    "mutation_rate": [0.01, 0.05, 0.1],
    "crossover_rate": [0.5, 0.7, 0.9]
}

# Hyperparameter tuning function
def tune_hyperparameters() -> List:
    # You should decide/engineer the `score` youself, which is the tuning objective
    best_score = float('inf')
    best_params = None
    # create the LABS problem and the data logger
    F18, _logger_18 = create_problem(dimension=50, fid=18)
    # create the N-Queens problem and the data logger
    F23, _logger_23 = create_problem(dimension=49, fid=23)
    
    scores = []
    params_dic = []
        
    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:
                # You should initialize you GA implementation with a hyperparameter setting
                # and execute it on both problems F18, and F23
                # please decide how many function evaluations you wish to use for running the GA
                # on each problem per each hyperparameter setting
                #......
                param_dic = dict.fromkeys(['pop_size', 'mutation_rate', 'crossover_rate'])

                current_best = float('inf')
                for _ in range(5): 
                    score = GA_evaluation(F18, F23, pop_size, mutation_rate, crossover_rate) 
                    if score<current_best:
                        current_best = score
                        param_dic['pop_size'] = pop_size
                        param_dic['mutation_rate'] = mutation_rate
                        param_dic['crossover_rate'] = crossover_rate
                
                scores.append(current_best)
                params_dic.append(param_dic)

    # inital points
    x0 = [[param['pop_size'], param['mutation_rate'], param['crossover_rate']] for param in params_dic]

    result = gp_minimize(func=lambda params: -GA_evaluation(F18, F23, *params), 
                         dimensions=[(10, 200), (0.001, 0.1), (0.5, 0.9)], 
                         n_calls=30, x0=x0, y0=scores)
    best_params = result.x
    
    '''if current_best<best_score:
                    best_score = current_best
                    best_params = result_dic'''
    
    # best_params = params_dic[scores.index(max(scores))]
    print(best_params)
    # return best_params

# ********** how to pass parameters to GA
def GA_evaluation(problem1, problem2, pop_size, mutation_rate, crossover_rate):
        
    GA_1(problem1, pop_size, mutation_rate, crossover_rate)
    # print(problem1.state.current_best)
    F_18 = problem1.state.current_best.y
    problem1.reset()
    GA_1(problem2, pop_size, mutation_rate, crossover_rate)
    F_23 = problem2.state.current_best.y
    # print(problem2.state.current_best)
    problem2.reset()
    score = (F_18 + F_23)/2
    return score


if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    #population_size, mutation_rate, crossover_rate = 
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    '''print(population_size)
    print(mutation_rate)
    print(crossover_rate)'''