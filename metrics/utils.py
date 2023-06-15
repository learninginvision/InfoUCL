# Fixes issue with matplotlib on linux server
import os
import numpy as np
from utils.logger import Simple_Logger


# This function is adapted from https://github.com/divyam3897/UCL/utils/metrics.py
def backward_transfer(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


# This function is adapted from https://github.com/divyam3897/UCL/utils/metrics.py
def forgetting(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


def save_results(data, save_path=None):
    '''Save test metric to file'''
    logger = Simple_Logger(save_path)
    
    # multi-task
    if len(data) == 1:
        logger.log('acc', data)
        logger.log('Average accuracy', np.mean(data))
        logger.log('Forgetting', 0.0)
        logger.log('Backward_Transfer', 0.0)
        
        return 
        
    # save test metric
    assert len(data) == len(data[-1]), 'The number of tasks is not equal to the number of test metric'
    logger.log('acc', data)
    logger.log('Average accuracy', np.mean(data[-1]))
    logger.log('Forgetting', forgetting(data))
    logger.log('Backward_Transfer', backward_transfer(data))