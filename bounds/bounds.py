import os, sys
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras import backend as K

import pandas as pd
import pickle
import gc, re, copy
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

# Project imports 
from data import mnist_m as mnistm
from data import mnist
from data.label_shift import label_shift_linear, plot_labeldist, plot_splitbars
from data.tasks import load_task
from experiments.training import *
from experiments.SL_bound import *
from experiments.DA_bound import *
from util.kl import *
from util.misc import *
from results.plotting import *

def compute_bound_parts(task, posterior_path, x_bound, y_bound, x_target, y_target, alpha=0.1, delta=0.05, epsilon=0.01, 
                  prior_path=None, bound='germain', binary=False, n_classifiers=4, sigma=[3,3], seed=None):

    print('\n'+'-'*40)
    print('Computing bound components for')
    print('   Prior: %s' % prior_path)
    print('   Posterior: %s' % posterior_path)
    print('Clearing session...')
    K.clear_session()
    
    print('Initializing models...')
    if binary:
        M_prior = init_MNIST_model_binary()
        M_posterior = init_MNIST_model_binary()
    else:
        M_prior = init_MNIST_model()
        M_posterior = init_MNIST_model()                

    # @TODO: Are the parameters for optimizer etc necessary when just loading the model?
    M_prior.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
    
    M_posterior.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
    
    ### load the prior weights if there are any
    if(binary and alpha != 0):
        prior_path="priors/"+"task"+str(task)+"/Binary/"+str(int(100*alpha))+"/prior.ckpt"
    elif(alpha != 0):
        prior_path="priors/"+"task"+str(task)+"/"+str(int(100*alpha))+"/prior.ckpt"
        
    print('Loading weights...')
    if alpha==0:
        ### do nothing, just take the random initialisation
        w_a=M_prior.get_weights()
    else:
        M_prior.load_weights(prior_path)
        w_a=M_prior.get_weights()
        
    # Load posterior weights
    M_posterior.load_weights(posterior_path)
    w_s=M_posterior.get_weights()
    
    t = time.time()
    
    ## do X draws of the posterior, for two separate classifiers
    sigma_tmp=sigma
    sigma=sigma[0]*10**(-1*sigma[1])
    
    print('Drawing classifiers...')
    w_s_draws = draw_classifier(w_s, sigma, num_classifiers=n_classifiers)
    w_s_draws2 = draw_classifier(w_s, sigma, num_classifiers=n_classifiers)
    
    elapsed = time.time() - t
    print('Time spent drawing the classifiers: %.4fs' % elapsed)
    
    """
    Calculate train and target errors
    """
    
    errorsum=[]
    target_errorsum=[]

    y_bound = np.array(y_bound)
    y_target = np.array(y_target)

    ######## in here we should make the results save in a vector for each part to be able to calculate
    ######## the standard deviation and be able to get error bars on things.
    print('Calculating errors...')
    t = time.time()
    for h in w_s_draws:
        M_posterior.set_weights(h)
        errorsum.append((1-M_posterior.evaluate(x_bound,y_bound,verbose=0)[1]))
        target_errorsum.append((1-M_posterior.evaluate(x_target,y_target,verbose=0)[1]))

    for hprime in w_s_draws2:
        M_posterior.set_weights(hprime)
        errorsum.append((1-M_posterior.evaluate(x_bound,y_bound,verbose=0)[1]))
        target_errorsum.append((1-M_posterior.evaluate(x_target,y_target,verbose=0)[1]))

    train_germain = np.mean(errorsum) 
    target_germain = np.mean(target_errorsum)  
    error_std = np.std(errorsum)
    target_error_std = np.std(target_errorsum)
    elapsed = time.time() - t
    print('Time spent calculating errors: %.4fs' % elapsed)
    
    
    """
    Calculate joint errors
    @TODO: This part should be merged with the above. Errors can be readibly computed from the predictions
    """
    
    e_ssum=[]
    e_tsum=[]
    d_txsum=[]
    d_sxsum=[]
    d_tx_h=0
    d_sx_h=0
    d_tx_hprime=0
    d_sx_hprime=0

    t = time.time()

    #### Here we just do the four pairs so there is no cross-usage
    #### this can be not good for the independence of the values which makes the CI useless

    print('Computing joint errors and disagreements...')
    for i, h in enumerate(w_s_draws):
        M_posterior.set_weights(h)
        d_tx_h=M_posterior.predict(x_target,verbose=0)
        d_sx_h=M_posterior.predict(x_bound,verbose=0)
        d_sx_h=make_01(d_sx_h)
        d_tx_h=make_01(d_tx_h)

        hprime=w_s_draws2[i]
        M_posterior.set_weights(hprime)
        d_tx_hprime=M_posterior.predict(x_target,verbose=0)
        d_sx_hprime=M_posterior.predict(x_bound,verbose=0)
        d_sx_hprime=make_01(d_sx_hprime)
        d_tx_hprime=make_01(d_tx_hprime)

        e_ssum.append(joint_error(d_sx_h,d_sx_hprime,y_bound))
        d_sxsum.append(classifier_disagreement(d_sx_h,d_sx_hprime))
        e_tsum.append(joint_error(d_tx_h,d_tx_hprime,y_target))
        d_txsum.append(classifier_disagreement(d_tx_h,d_tx_hprime))

        
    # Means
    e_s = np.mean(e_ssum)
    d_sx = np.mean(d_sxsum)
    e_t = np.mean(e_tsum)
    d_tx = np.mean(d_txsum)
    
    # Stds
    e_s_std = np.std(e_ssum)
    d_sx_std = np.std(d_sxsum)
    e_t_std = np.std(e_tsum)
    d_tx_std = np.std(d_txsum)
    
    elapsed = time.time() - t
    print("Time spent calculating joint errors and disagreements: "+str(elapsed)+"\n")    


    """
    Compute the KL divergence
    """
    t = time.time()
    KL = estimate_KL(w_a, w_s, sigma) ## compute the KL

    elapsed = time.time() - t
    print("Time spent calculating KL: "+str(elapsed)+"\n") 
    

    print("Finished calculation of bound parts")
    
    """
    Finish up and store results
    """
    
    # Checkpoint corresponds to either update or epoch depending on first part 1_ or 2_ """
    checkpoint = os.path.splitext(os.path.basename(posterior_path))[0]
    
    updates = []
    if checkpoint[0:2]=="1_":
            updates = int(checkpoint[2:])
    else: 
        updates = (int(checkpoint[2:])+1)*547 # @TODO: Constant hack
       
    if binary:
        result_path="results/"+"task"+str(task)+"/Binary/"+str(int(1000*epsilon))+\
            "_"+str(int(100*alpha))+"_"+str(sigma_tmp[0])+str(sigma_tmp[1])+'_'+checkpoint+'_results.pkl'
    else:
        result_path="results/"+"task"+str(task)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+\
        "_"+str(sigma_tmp[0])+str(sigma_tmp[1])+'_'+checkpoint+'_results.pkl'
        
    # Create dir
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
    results=pd.DataFrame({
        'Weightupdates': [updates],
        'train_germain': [train_germain],
        'target_germain': [target_germain],
        'KL': [KL],
        'e_s': [e_s],
        'e_t': [e_t],
        'd_tx': [d_tx], 
        'd_sx': [d_sx],
        'error_std': [error_std],
        'target_error_std': [target_error_std],
        'e_s_std': [e_s_std],
        'e_t_std': [e_t_std],
        'd_tx_std': [d_tx_std],
        'd_sx_std': [d_sx_std], 
        'alpha': [alpha], 
        'sigma': [sigma], 
        'epsilon': [epsilon],
        'checkpoint': [checkpoint], 
        'seed': [seed]
    })
   
    print('Saving results in %s ...' % result_path)
    results.to_pickle(result_path)
    print('Done.')
    print('-'*40 + '\n')
    
    return results 

    """
    The reimaining part only makes sense in the context of a set of snapshots
    
    # Number of samples 
    m=len(y_bound)
    
     # calculate disrho bound
    [res,bestparam, boundparts]=grid_search(train_germain,e_s,e_t,d_tx,d_sx,KL,delta,m)
    
    # calculate beta bound
    [res2,bestparam2, boundparts2]=grid_search(train_germain,e_s,e_t,d_tx,d_sx,KL,delta,m,beta_bound=True)            
                
    results['germain_bound']=res
    print("Germain bound"+str(res))
    print("[a, omega]= "+str(bestparam))
    
    Best=np.zeros([len(res),1])
    Best[0]=bestparam[0]
    Best[1]=bestparam[1]
    Best[2]=CLASSIFIERS
    #print(Best)
    results['bestparam']=Best
    results['boundpart1_germain']=boundparts[0]
    results['boundpart2_germain']=boundparts[1]
    results['boundpart3_germain']=boundparts[2]
    results['boundpart4_germain']=boundparts[3]
    results['boundpart5_germain']=boundparts[4]
    ## beta bound
    results['beta_bound']=res2
    results['beta_boundpart1']=boundparts2[0]
    results['beta_boundpart2']=boundparts2[1]
    results['beta_boundpart3']=boundparts2[2]
    
    fpath = project_folder+'mnist_transfer/'+result_path+"_results.pkl"
    print('Saving into: %s' % fpath)
    
    with open(fpath,'wb') as f:
        pickle.dump(results,f)
    f.close()
    return results
    """