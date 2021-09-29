# System imports
import os, sys
import numpy as np
import argparse

# Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# Project imports
from data.tasks import load_task
from bounds.bounds import *
from util.misc import *
  
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compute bound parts.')
    
    parser.add_argument('-t', '--task', type=int, default=2, dest='task')
    parser.add_argument('-r', '--seed', type=int, default=69105, dest='seed')
    parser.add_argument('-a', '--alpha', type=float, default=0., dest='alpha')
    parser.add_argument('-s', '--sigma', type=str, default='3,3', dest='sigma')
    parser.add_argument('-e', '--epsilon', type=float, default=0., dest='epsilon')
    parser.add_argument('-d', '--delta', type=float, default=0.05, dest='delta')
    parser.add_argument('-b', '--binary', type=int, default=0, dest='binary')
    parser.add_argument('-n', '--nclassifiers', type=int, default=2, dest='n_classifiers')
    parser.add_argument('-B', '--bound', type=str, default='germain', dest='bound')
    parser.add_argument('-p', '--prior', type=str, default=None, dest='prior_path')
    parser.add_argument('-P', '--posterior', type=str, default='', dest='posterior_path')
    
    args = parser.parse_args()
    print(args.__dict__)
    
    seed = args.seed
    task = args.task
    alpha = args.alpha
    s = args.sigma.split('.')
    sigma = [int(s[0]), int(s[1])]
    epsilon = args.epsilon
    delta = args.delta
    binary = args.binary>0
    n_classifiers = args.n_classifiers
    bound = args.bound
    prior_path = args.prior_path
    posterior_path = args.posterior_path

    if prior_path == '': 
        prior_path = None
    if posterior_path == '': 
        raise Exception('No posterior path specified')
    
    ################################################################################

    print('Loading data...')
    x_source, y_source, x_target, y_target = load_task(task)
    
    print('Making data binary...')
    if binary: 
        y_target = make_mnist_binary(y_target)
        y_source = make_mnist_binary(y_source)
        
    if alpha==0:
        x_bound=x_source
        y_bound=y_source
    else:
        x_bound, x_prior, y_bound, y_prior = train_test_split(x_source, y_source, test_size=alpha, random_state=seed)
        
    print('\n'+'-'*40)
        
    results = compute_bound_parts(task, posterior_path, x_bound, y_bound, x_target, y_target, 
                        prior_path=prior_path, bound=bound, binary=binary, sigma=sigma, alpha=alpha,
                        delta=delta, epsilon=epsilon, n_classifiers=n_classifiers, seed=seed)
    checkpoint = results['checkpoint'].values.ravel()[0]
    
    if binary:
        result_path="results/"+"task"+str(task)+"/Binary/"+str(int(1000*epsilon))+\
            "_"+str(int(100*alpha))+"_"+str(sigma[0])+str(sigma[1])+'_'+checkpoint+'_results.pkl'
    else:
        result_path="results/"+"task"+str(task)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+\
        "_"+str(sigma[0])+str(sigma[1])+'_'+checkpoint+'_results.pkl'
        
    # Create dir
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    print('Saving results in %s ...' % result_path)
    results.to_pickle(result_path)
    print('Done.')
    print('-'*40 + '\n')
   