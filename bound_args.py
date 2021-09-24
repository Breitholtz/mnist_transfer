# imports
from __future__ import print_function 

#%load_ext autoreload
#%autoreload 2

from matplotlib import pyplot as plt
#%matplotlib inline

import numpy as np
import os, sys
import argparse
import time

import tensorflow as tf

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
from experiments.training import *
from experiments.SL_bound import *
from experiments.DA_bound import *
from util.kl import *
from util.misc import *
from results.plotting import *

# Hyper-parameters
batch_size = 128
num_classes = 10
epochs = 10
make_plots = False
delta=0.05 ## what would this be?   

TASK = 2
def read_weights_germain(model,w_a,x_bound,y_bound,x_target,y_target,epochs_trained,sigma,epsilon,alpha,Binary=False,Task=TASK):
    import re
    KLs=[]
    e_s=[]
    e_t=[]
    d_tx=[]
    d_sx=[]
    epochs=[]
    train_germain=[] 
    target_germain=[]
    dis_rho=[]
    lambda_rho=[]
    sigma_tmp=sigma
    sigma=sigma[0]*10**(-1*sigma[1])
    
    ### Here we do something more intelligent to not have to hardcode the epoch amounts. 
    ### we parse the filenames and sort them in numerical order and then load the weights
    if Binary:
        path="posteriors/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
    else:
        path="posteriors/"+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
    list1=[]
    list2=[]
    dirFiles = os.listdir(path) #list of directory files
    ## remove the ckpt.index and sort so that we get the epochs that are in the directory
    for files in dirFiles: #filter out all non jpgs
        if '.ckpt.index' in files:
            name = re.sub('\.ckpt.index$', '', files)
            ### if it has a one it goes in one list and if it starts with a two it goes in the other
            if (name[0]=="1"):
                list1.append(name)
            elif (name[0]=="2"):
                list2.append(name)
            #epochs.append(name)
    #epochs.sort(key=lambda f: int(re.sub('\D', '', f)))
    list1.sort(key=lambda f: int(re.sub('\D', '', f)))
    num_batchweights=len(list1)
    list2.sort(key=lambda f: int(re.sub('\D', '', f)))
    list1.extend(list2)
    Ws=list1 ## vector of checkpoint filenames
    #print(Ws)
    #sys.exit(-1)
    #epochs=[int(i) for i in Ws]
    Xvector=[]
    for i in Ws:
        #print(i)
        if i[0]=="1":
            if i[1]=="_":
                Xvector.append(int(i[2:]))
    for i in list2:
        Xvector.append((int(i[2:])+1)*547) ## TODO contant hack
    print(Xvector)
    #sys.exit(-1)
    """   
    epochs = [] #list of checkpoint filenames
    dirFiles = os.listdir(path) #list of directory files
    ## remove the ckpt.index and sort so that we get the epochs that are in the directory
    for files in dirFiles: #filter out all non jpgs
        if '.ckpt.index' in files:
            name = re.sub('\.ckpt.index$', '', files)
            epochs.append(name)
    epochs.sort(key=lambda f: int(re.sub('\D', '', f)))
    epochs=[int(i) for i in epochs]
    """ 
    path="posteriors/"+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"/"#"{epoch:0d}.ckpt"
    if Binary:
        path="posteriors/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"/"#"{epoch:0d}.ckpt"
    
    
    L=len(Xvector)
     ### vectors for saving std of each point
    error_std=[]
    target_error_std=[]
    e_s_std=[]
    e_t_std=[]
    d_tx_std=[]
    d_sx_std=[]
    for checkpoint in Ws:
        if Binary:
            model=init_MNIST_model_binary()
            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
        else:
            model=init_MNIST_model()
            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)

        model.load_weights(path+str(checkpoint)+".ckpt")
        w_s=model.get_weights()
        """
        ############################# test
        #print(x_target[1])
        pred = model.predict(x_target)
        print(make_01(pred))
        indices = [i for i,v in enumerate(pred) if np.sum(pred[i]-y_target[i])==0]
        print(indices)
        sys.exit(-1)
        """
        ##### here we should draw classifiers and pass that on
        CLASSIFIERS=4
        t = time.time()
        ## do X draws of the posterior, for two separate classifiers
        w_s_draws=draw_classifier(w_s,sigma,num_classifiers=CLASSIFIERS)
        w_s_draws2=draw_classifier(w_s,sigma,num_classifiers=CLASSIFIERS)
        ## do X draws of the prior
        #w_a_draws=draw_classifier(w_a,sigma,num_classifiers=2)
          
        elapsed = time.time() - t
        print("Time spent drawing the classifiers: "+str(elapsed)+"\n")
   
        
        ###### for each pair of drawn prior and posterior we calculate the necessary parts of the bound 
        ###### and then average the result and return that
        
        

        errorsum=[]
        target_errorsum=[]
        
        #
        # @TODO: Fredrik hack
        #
        y_bound = np.array(y_bound)
        y_target = np.array(y_target)
       
        ######## in here we should make the results save in a vector for each part to be able to calculate
        ######## the standard deviation and be able to get error bars on things.
        t = time.time()
        for h in w_s_draws:
            model.set_weights(h)
            errorsum.append((1-model.evaluate(x_bound,y_bound,verbose=0)[1]))
            target_errorsum.append((1-model.evaluate(x_target,y_target,verbose=0)[1]))
        
        
        for hprime in w_s_draws2:
            model.set_weights(hprime)
            errorsum.append((1-model.evaluate(x_bound,y_bound,verbose=0)[1]))
            target_errorsum.append((1-model.evaluate(x_target,y_target,verbose=0)[1]))
     
        
        train_germain.append((np.mean(errorsum))) #/(len(w_s_draws)+len(w_s_draws2))
        target_germain.append(np.mean(target_errorsum))  #/(len(w_s_draws)+len(w_s_draws2))
        error_std.append(np.std(errorsum))
        target_error_std.append(np.std(target_errorsum))
        elapsed = time.time() - t
        print("Time spent calculating errors: "+str(elapsed)+"\n")
        
        ######## in here we should make the results save in a vector for each part to be able to calculate
        ######## the standard deviation and be able to get error bars on things.
        
        #### loop over pairs of classifiers from posterior for the disagreement and joint error
        #q=0
        e_ssum=[]
        e_tsum=[]
        d_txsum=[]
        d_sxsum=[]
        d_tx_h=0
        d_sx_h=0
        d_tx_hprime=0
        d_sx_hprime=0
       
        t = time.time()
        
        #### Here we should just do the four pairs so there is no cross-usage here
        #### this can be not good for the independence of the values which makes the CI useless
        
        for i, h in enumerate(w_s_draws):
            model.set_weights(h)
            d_tx_h=model.predict(x_target,verbose=0)
            d_sx_h=model.predict(x_bound,verbose=0)
            d_sx_h=make_01(d_sx_h)
            d_tx_h=make_01(d_tx_h)
            #for hprime in w_s_draws2:
            hprime=w_s_draws2[i]
            model.set_weights(hprime)
            d_tx_hprime=model.predict(x_target,verbose=0)
            d_sx_hprime=model.predict(x_bound,verbose=0)
            d_sx_hprime=make_01(d_sx_hprime)
            d_tx_hprime=make_01(d_tx_hprime)
                
            e_ssum.append(joint_error(d_sx_h,d_sx_hprime,y_bound))
            d_sxsum=(classifier_disagreement(d_sx_h,d_sx_hprime))
            e_tsum=(joint_error(d_tx_h,d_tx_hprime,y_target))
            d_txsum=(classifier_disagreement(d_tx_h,d_tx_hprime))
        
        e_s.append(np.mean(e_ssum))
        d_sx.append(np.mean(d_sxsum))
        e_t.append(np.mean(e_tsum))
        d_tx.append(np.mean(d_txsum))
        ### save the std
        e_s_std.append(np.std(e_ssum))
        d_sx_std.append(np.std(d_sxsum))
        e_t_std.append(np.std(e_tsum))
        d_tx_std.append(np.std(d_txsum))
        elapsed = time.time() - t
        print("Time spent calculating joint errors and disagreements: "+str(elapsed)+"\n")    
        
        KLsum=0
        t = time.time()
        
        KLsum=estimate_KL(w_a,w_s,sigma)## compute the KL
        ## only w_s, w_a here
                
      
        KLs.append(KLsum)
        elapsed = time.time() - t
        print("Time spent calculating KL: "+str(elapsed)+"\n") 
        
        ### memory leak city
        del model
        _=gc.collect()
        
        
     
    print("Finished calculation of bound parts")
   
          #### load the result file if it exists otherwise make one
    if Binary:
        result_path="results/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"+str(sigma_tmp[0])+str(sigma_tmp[1])
    else:
        result_path="results/"+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"+str(sigma_tmp[0])+str(sigma_tmp[1])
    if(os.path.exists(result_path+"_results.pkl")):
        results=pd.read_pickle(result_path+"_results.pkl")
    else:
        results=pd.DataFrame({'Weightupdates': Xvector,
        'train_germain': train_germain,
        'target_germain':target_germain,
        'KL': KLs})
        with open(path_to_root_file+'mnist_transfer/'+result_path+"_results.pkl",'wb') as f:
            pickle.dump(results,f)
        f.close()
        results=pd.read_pickle(result_path+"_results.pkl")

    train_germain=np.array(train_germain)
    results['Weightupdates']=Xvector
    results['train_germain']=train_germain
    results['target_germain']=target_germain
    results['e_s']=e_s
    results['e_t']=e_t
    results['d_tx']=d_tx
    results['d_sx']=d_sx
    results['KL']=KLs
    ### save the std deviations as well in some form like a vector of deviations for each factor
    results['error_std']=error_std
    results['target_error_std']=target_error_std
    results['e_s_std']=e_s_std
    results['e_t_std']=e_t_std
    results['d_tx_std']=d_tx_std
    results['d_sx_std']=d_sx_std
    KL=KLs
    
    m=len(y_bound)
    delta=0.05 ## hardcoded value
     # calculate disrho bound
    [res,bestparam, boundparts]=grid_search(train_germain,e_s,e_t,d_tx,d_sx,KL,delta,m,L)
    # calculate beta bound
    [res2,bestparam2, boundparts2]=grid_search(train_germain,e_s,e_t,d_tx,d_sx,KL,delta,m,L,beta_bound=True)            
                
                
                
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
    with open(path_to_root_file+'mnist_transfer/'+result_path+"_results.pkl",'wb') as f:
        pickle.dump(results,f)
    f.close()
    return results

def grid_search(train_germain,e_s,e_t,d_tx,d_sx,KL,delta,m,L,beta_bound=False):
    #### here we want to do a coarse grid search over a and omega to get the smallest bound 
    print("Starting gridsearch....")
    avec=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000]
    omegas=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000]
    tmp= sys.maxsize
    res=[]
    bestparam=[0,0]
    for a in avec:
        for omega in omegas:
            if beta_bound:
                germain_bound, boundparts=calculate_beta_bound(e_s,d_tx,KL,delta,a,omega,m,L)
            else:
                germain_bound,boundparts=calculate_germain_bound(train_germain,e_s,e_t,d_tx,d_sx,KL,delta,a,omega,m,L)
            if min(germain_bound)<tmp:
                tmp=min(germain_bound)
                #print("Best bound thus far:"+str(tmp))
                res=germain_bound
                bestparam=[a,omega]
                
    ### do a finer sweep around the best parameters
    if bestparam[0]!=0:
        avec=np.arange(bestparam[0]-bestparam[0]/2,bestparam[0]+bestparam[0]*4,0.1*bestparam[0])
    else:## no bound better than the max int was found, if that is even possible
        avec=np.arange(-1,1,0.1)
    if bestparam[1]!=0:
        omegas=np.arange(bestparam[1]-bestparam[1]/2,bestparam[1]+bestparam[1]*4,0.1*bestparam[1])
    else:## no bound better than the max int was found
        avec=np.arange(-1,1,0.1)
    boundparts=[0, 0,0,0,0]
    for a in avec:
        for omega in omegas:
            if beta_bound:
                germain_bound, boundparts=calculate_beta_bound(e_s,d_tx,KL,delta,a,omega,m,L)
            else:
                germain_bound,boundparts=calculate_germain_bound(train_germain,e_s,e_t,d_tx,d_sx,KL,delta,a,omega,m,L)
            if min(germain_bound)<tmp:
                tmp=min(germain_bound)
                #print("Best finer bound thus far:"+str(tmp))
                res=germain_bound
                bestparam=[a,omega]
                #boundparts=[a1,a2,a3,a4,a5]
    return [res,bestparam, boundparts]
def calculate_beta_bound(e_s,d_tx,KL,delta,b,c,m,L,BETA=0):
    BETA=10.986111 ### hardcoded value for beta_infinity for TASK2 TODO!
    m_s=m  ## temporary, we should pass these in
    m_t=m  ## temporary, we should pass these in
    bprime=b/(1-np.exp(-b))
    cprime=c/(1-np.exp(-c))
    
    bound=[]
    a1=np.zeros(L)
    a2=np.zeros(L)
    a3=np.zeros(L)
    for i in range(L):
        a1[i]=cprime/2*(d_tx[i])
        a2[i]=bprime*e_s[i]
        a3[i]=(cprime/(m_t*c)+bprime*BETA/(m_s*b))*(2*KL[i]+np.log(2/delta))
    ## we cannot evaluate the eta term in the bound so this is it. For TASK 2 it is 0 anyway.
    ## And for other tasks we will not evaluate this bound anyway as we probably have no way of doing so easily..
        bound.append(a1[i]+a2[i]+a3[i])
    boundparts=[a1,a2,a3]
    return bound, boundparts

def germain_bound(x_bound,y_bound,x_target,y_target,alpha,sigma,epsilon,task,Binary=False):
    TASK=task
    sigma_tmp=sigma
    sigma=sigma[0]*10**(-1*sigma[1])
        ## read params.txt for the desired alpha and get the parameters
 
    if Binary:
        with open('posteriors/'+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+'/params.txt', 'rb+') as f:
            params=f.readlines()
        f.close()
        prior_path="priors/"+"task"+str(TASK)+"/Binary/"+str(int(100*alpha))+"/prior.ckpt"
        result_path="results/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"
    else:
        with open('posteriors/'+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+'/params.txt', 'rb+') as f:
            params=f.readlines()
        f.close()
        prior_path="priors/"+"task"+str(TASK)+"/"+str(int(100*alpha))+"/prior.ckpt"
        result_path="results/"+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"

    epsilon=float(params[1])
    epochs_trained=int(params[2])
    if Binary:
        M=init_MNIST_model_binary()
    else:
        M=init_MNIST_model()
                
    M.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
      ### load the prior weights if there are any
    if(Binary and alpha != 0):
        prior_path="priors/"+"task"+str(TASK)+"/Binary/"+str(int(100*alpha))+"/prior.ckpt"
    elif(alpha != 0):
        prior_path="priors/"+"task"+str(TASK)+"/"+str(int(100*alpha))+"/prior.ckpt"
    if alpha==0:
        ### do nothing, just take the random initialisation
        w_a=M.get_weights()
    else:
        M.load_weights(prior_path)
        w_a=M.get_weights()
    
 
    ## get the prior weights for the KL and pass into read_weights
    results=read_weights_germain(M,w_a,x_bound,y_bound,x_target,y_target,epochs_trained,sigma_tmp,epsilon,alpha,Binary=Binary,Task=TASK)
    return results
def main(seed=0, batch_size=32, alpha=0.1, epsilon=0.01,sigma=[3,2], task=2):

    np.random.seed(seed)

    """ Data loading """
    if task==2:
        x_train, y_train, x_test, y_test = mnist.load_mnist()
        x_train_m, y_train_m, x_test_m, y_test_m = mnistm.load_mnistm(y_train,y_test)
        ###### Add train and test together and shift the distributions to create source and target distributions
        ### MNIST all data
        x_full=np.append(x_train,x_test, axis=0)
        y_full=np.append(y_train,y_test, axis=0)
        ### MNIST-M all data
        x_full_m=np.append(x_train_m,x_test_m, axis=0)
        y_full_m=np.append(y_train_m,y_test_m, axis=0)
        #x_shift,y_shift,x_shift_target,y_shift_target =label_shift(x_train,y_train,1/2,7)
        x_shift, y_shift, x_shift_target, y_shift_target = label_shift_linear(x_full,y_full,1/12,[0,1,2,3,4,5,6,7,8,9])
        x_shift_m, y_shift_m,x_shift_target_m, y_shift_target_m = label_shift_linear(x_full_m,y_full_m,1/12,[0,1,2,3,4,5,6,7,8,9],decreasing=False)
        ##### calculate the label densities here
        densities=[]
        densities.append(np.sum(y_shift,axis=0))
        densities.append(np.sum(y_shift_m,axis=0))
        densities.append(np.sum(y_shift_target,axis=0))
        densities.append(np.sum(y_shift_target_m,axis=0))
        
        
        L=len(densities[0])
        interdomain_densities = [[] for x in range(2)]
        for i in range(L):
            ## all densities are # in mnist over # in mnist-m
            interdomain_densities[0].append(densities[0][i]/densities[1][i])
            interdomain_densities[1].append(densities[2][i]/densities[3][i])
        print(interdomain_densities)
        x_source=np.append(x_shift,x_shift_m, axis=0)
        y_source=np.append(y_shift,y_shift_m, axis=0)
        x_target=np.append(x_shift_target,x_shift_target_m, axis=0)
        y_target=np.append(y_shift_target,y_shift_target_m, axis=0)
        #""" Bound computation """
    
    Binary=True ## set this from argparse?
    y_target_bin=make_mnist_binary(y_target)
    y_source_bin=make_mnist_binary(y_source)


    print("alpha:"+str(alpha))
    if alpha==0:
        x_bound=x_source
        y_bound=y_source_bin
    else:
        x_bound, x_prior, y_bound , y_prior = train_test_split(x_source,y_source_bin,test_size=alpha,random_state=69105)

    print("epsilon:"+str(epsilon))

    print("sigma:"+str(sigma))
    results=germain_bound(x_bound,y_bound,x_target,y_target_bin,alpha=alpha,
                              sigma=sigma,epsilon=epsilon,task=task,Binary=Binary)

    print(seed)
    print(task)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate bounds from saved weights.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, dest='batch_size')
    parser.add_argument('-r', '--seed', type=int, default=0, dest='seed')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, dest='alpha')
    parser.add_argument('-e', '--epsilon', type=float, default=0.01, dest='epsilon')
    parser.add_argument('-s', '--sigma', nargs='+', type=int , default="3 2", dest='sigma')
    parser.add_argument('-t', '--task', type=int, default=2, dest='task')


    args = parser.parse_args()

    main(batch_size=int(args.batch_size), seed=int(args.seed), alpha=float(args.alpha),epsilon=float(args.epsilon), sigma=args.sigma,task=int(args.task))
