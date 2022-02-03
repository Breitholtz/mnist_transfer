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
from experiments.training import *
from util.misc import *
project_folder="/cephyr/users/adambre/Alvis/mnist_transfer/"
#project_folder="/cephyr/NOBACKUP/groups/snic2021-23-538/mnist_transfer/"
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train and save prior and posterior for one set of parameters.')
    
    parser.add_argument('-t', '--task', type=int, default=2, dest='task')
    parser.add_argument('-r', '--seed', type=int, default=69105, dest='seed')
    parser.add_argument('-a', '--alpha', type=float, default=0., dest='alpha')
    parser.add_argument('-e', '--epsilon', type=float, default=0., dest='epsilon')
    parser.add_argument('-d', '--delta', type=float, default=0.05, dest='delta')
    parser.add_argument('-b', '--binary', type=int, default=0, dest='binary')
    parser.add_argument('-A', '--architecture', type=str, default='lenet', dest='architecture')
    
    args = parser.parse_args()
    print(args.__dict__)
    
    seed = args.seed
    task = args.task
    alpha = args.alpha
    epsilon = args.epsilon
    delta = args.delta
    binary = args.binary>0
    architecture=args.architecture
    
    ################################################################################

    print('Loading data...')
    x_source, y_source, x_target, y_target = load_task(task,binary)
    
        
    if alpha==0:
        pass
    else:
        x_bound, x_prior, y_bound, y_prior = train_test_split(x_source, y_source, test_size=alpha, random_state=seed)
        w_a=train_prior(alpha,1,x_prior,y_prior,x_target=x_target,y_target=y_target,
                        save=True,task=task,binary=binary,batch_size=128,architecture=architecture)
        
    print('\n'+'-'*40)
    """
    Here we train the posterior
    """
    print('Training posterior with alpha=%f, epsilon=%f ...' % (alpha, epsilon))
    print('-'*40 + '\n')
    w_s=train_posterior(alpha,x_source,y_source,None,x_test=x_source,y_test=y_source,
                        save=True,epsilon=epsilon,task=task,binary=binary,batch_size=128,architecture=architecture)

    print('Done.')
    print('-'*40 + '\n')
   