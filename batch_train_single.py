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
#project_folder="/cephyr/users/adambre/Alvis/mnist_transfer/"
project_folder="/cephyr/NOBACKUP/groups/snic2021-23-538/mnist_transfer/"
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train and save prior and posterior for one set of parameters.')
    
    parser.add_argument('-t', '--task', type=int, default=2, dest='task')
    parser.add_argument('-r', '--seed', type=int, default=69105, dest='seed')
    parser.add_argument('-a', '--alpha', type=float, default=0., dest='alpha')
    #parser.add_argument('-e', '--epsilon', type=float, default=0., dest='epsilon')
    parser.add_argument('-d', '--delta', type=float, default=0.05, dest='delta')
    parser.add_argument('-b', '--binary', type=int, default=0, dest='binary')
    parser.add_argument('-B', '--batch_size', type=int, default=128, dest='batch_size')
    parser.add_argument('-A', '--architecture', type=str, default='lenet', dest='architecture')
    parser.add_argument('-I', '--image_size', type=int, default=32, dest='image_size')
    args = parser.parse_args()
    print(args.__dict__)
    
    seed = args.seed
    task = args.task
    alpha = args.alpha
    #epsilon = args.epsilon
    image_size = args.image_size
    delta = args.delta
    binary = args.binary>0
    architecture=args.architecture
    batch_size=args.batch_size
    ################################################################################

    print('Loading data...')
    
    if task==7:
        if alpha==0:
            source_generator, target_generator=load_task(task=task,alpha=alpha,architecture=architecture,
                                                         binary=True,image_size=image_size,seed=seed)
        else:
            prior_generator, bound_generator, target_generator=load_task(task=task,alpha=alpha,architecture=architecture,
                                                                         binary=True,image_size=image_size,seed=seed)
            
    else:
        x_source, y_source, x_target, y_target,_,_ = load_task(task,binary,architecture=architecture,image_size=image_size,seed=seed)
        
    if alpha==0:
        pass
    else:
        if task==7:
            w_a=train_prior(alpha,1,generator=prior_generator,val_generator=target_generator,
                            save=True,task=task,binary=True,batch_size=batch_size,architecture=architecture,
                            seed=seed,image_size=image_size)
        else:
            x_bound, x_prior, y_bound, y_prior = train_test_split(x_source, y_source, test_size=alpha, random_state=seed)
            w_a=train_prior(alpha,1,x_train=x_bound,y_train=y_bound,x_target=x_target,y_target=y_target,
                            save=True,task=task,binary=binary,batch_size=batch_size,architecture=architecture,seed=seed,image_size=image_size)
    print('\n'+'-'*40)
    """
    Here we train the posterior
    """
    print('Training posterior with alpha=%f...' % (alpha))
    print('-'*40 + '\n')
    if task==7:
        if alpha!=0:
            source_generator, target_generator=load_task(task=task,alpha=0,architecture=architecture,
                                                         binary=True,image_size=image_size,seed=seed)
        w_s=train_posterior(alpha,generator=source_generator,val_generator=target_generator,
                            prior_weights=None,save=True,task=task,binary=True,batch_size=batch_size,
                            architecture=architecture,seed=seed,image_size=image_size)
    else:
        w_s=train_posterior(alpha,x_train=x_source,y_train=y_source,prior_weights=None,x_test=x_source,y_test=y_source,
                        save=True,task=task,binary=binary,batch_size=batch_size,architecture=architecture,seed=seed,image_size=image_size)


    print('Done.')
    print('-'*40 + '\n')
   