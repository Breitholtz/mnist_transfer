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
from data.tasks import *
from experiments.training import *
#from experiments.SL_bound import *
#from experiments.DA_bound import *
from util.kl import *
from util.misc import *
from results.plotting import *

project_folder2="/cephyr/users/adambre/Alvis/"
project_folder="/cephyr/NOBACKUP/groups/snic2021-23-538/mnist_transfer/"

def draw_classifier(weights,sigma,num_classifiers):
    """
    Takes in the weights of a network along with a variance parameter and 
    samples num_classifiers many diffferent weights from a gaussian centered around the initial weights
    """
    w_s_draw=[[]]*num_classifiers
    ## draw new classifiers
    for draw in range(len(w_s_draw)):
        # for each weight matrix draw new normally distributed weights
        L=len(weights)
        w_tmp=weights.copy()
        for i in range(L):#weights: ### flatten, draw, reshape
            if(w_tmp[i].ndim>1):
                shapes=w_tmp[i].shape
                a=w_tmp[i].flatten()
                add=np.random.randn(len(a))*sigma
                new=a+add
                new=np.reshape(new,shapes)
                w_tmp[i]=new
            else:
                add=np.random.randn(len(w_tmp[i]))*sigma
                w_tmp[i]=w_tmp[i]+add               
        w_s_draw[draw]=w_tmp
    return w_s_draw

def joint_error(prediction_h,prediction_hprime,true_label):
    """
    This computes the emp.expected joint error, i.e. we approximate e_S= E_h,h' E_x,y L(h(x),y)L(h'(x),y)    
    """
    ## expected joint error
    shapes=prediction_h.shape
    e_s=0
    # e_S= E_h,h' E_x,y L(h(x),y)L(h'(x),y)     
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            e_s+=(prediction_h[i][j]-true_label[i][j])*(prediction_hprime[i][j]-true_label[i][j])
    e_s/=(2*shapes[0])
    return e_s

def classifier_disagreement(prediction_h,prediction_hprime):
    """
    This computes the emp.expected classifier disagreement, i.e. we R(h,h')= 1/n sum(L( h(x),h'(x) ))
    """
    shapes=prediction_h.shape
    d=0
    arr=np.abs(prediction_h-prediction_hprime)
    for i in arr:
        if np.sum(i)==2:
            d+=1
    d/=(shapes[0])
    return d
def calculate_germain_bound(train_error,e_s,e_t,d_tx,d_sx, KL,delta,a,omega,m,L):
    """
    Takes in parts and puts them together into the additive disrho bound from (Germain et al. 2013)
    """
    bound=[]
    aprime=2*a/(1-np.exp(-2*a))
    omegaprime=omega/(1-np.exp(-omega))
    a1=np.zeros(L)
    a2=np.zeros(L)
    a3=np.zeros(L)
    a4=np.zeros(L)
    a5=np.zeros(L)
    for i in range(L):
        lambda_rho=np.abs(e_t[i]-e_s[i])
        dis_rho=np.abs(d_tx[i]-d_sx[i])
        a1[i]=omegaprime*train_error[i]
        a2[i]=aprime/2*(dis_rho)
        a3[i]=(omegaprime/omega+aprime/a)*(KL[i]+np.log(3/delta))/m
        a4[i]=lambda_rho
        a5[i]=(aprime-1)/2
        bound.append(a1[i]+a2[i]+a3[i]+a4[i]+a5[i])
    #print(bound)
    return bound,a1,a2,a3,a4,a5


def make_01(predictions):
    """
    takes in non integer predictions and returns 1 for the most likely prediction and 0 for the others
    """
    new_predictions=np.zeros(predictions.shape)
    for i, row in enumerate(predictions):
        idx = np.where(row == np.amax(row))
        row=np.zeros(row.shape)
        row[idx]=1
        new_predictions[i]=row
    return new_predictions




def error_from_prediction(pred,y):
        ### note that this is for binary classification, when multiclass is used you have to change the 2 in the denominator to num_classes
    pred=make_01(pred)
    length=len(pred)
    return np.sum(np.abs(pred-y))/(2*length)
def compute_bound_parts(task, posterior_path, x_bound, y_bound, x_target, y_target, alpha=0.1, delta=0.05, epsilon=0.01, 
                  prior_path=None, bound='germain', binary=False, n_classifiers=4, sigma=[3,3], seed=None,batch_size=128,architecture="lenet"):
    posterior_path=posterior_path
    print('Computing bound components for')
    print('   Prior: %s' % prior_path)
    print('   Posterior: %s' % posterior_path)
    print('Clearing session...')
    K.clear_session()
    
    print('Initializing models...')

    M_prior=init_task_model(task,binary,architecture) 
    M_posterior=M_prior
     # @TODO: Are the parameters for optimizer etc necessary when just loading the model?
    M_prior.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
    
    M_posterior.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
    
    ### load the prior weights if there are any
    if(binary and alpha != 0):
        prior_path=project_folder+"priors/"+"task"+str(task)+"/Binary/"+str(architecture)+"/"+str(int(100*alpha))+"/prior.ckpt"
    elif(alpha != 0):
        prior_path=project_folder+"priors/"+"task"+str(task)+"/"+str(architecture)+"/"+str(int(100*alpha))+"/prior.ckpt"
        
    print('Loading weights...')
    if alpha==0 or prior_path is None:
        ### do nothing, just take the random initialisation
        w_a=M_prior.get_weights()
    else:
        M_prior.load_weights(prior_path)
        w_a=M_prior.get_weights()
        
    # Load posterior weights
    M_posterior.load_weights(posterior_path)
    w_s=M_posterior.get_weights()
    
  
    ## do X draws of the posterior, for two separate classifiers
    sigma_tmp=sigma
    sigma=sigma[0]*10**(-1*sigma[1])
    e_s, e_t, d_sx, d_tx, e_s_std, e_t_std, d_sx_std, d_tx_std, train_germain, target_germain, error_std, target_error_std=draw_classifier_and_calculate_errors(w_s,sigma,n_classifiers,x_bound,y_bound,x_target,y_target,M_posterior)


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
        #l=len(x_bound)
        if task==2:
            batch_num=547
        elif task==6:
            batch_num=1875
        else:
            print("error!!!!")
            batch_num=1
        #batch_num=np.ceil(l/batch_size) 
        #print(batch_num)
        #print(updates)
        updates = (int(checkpoint[2:])+1)*batch_num # constant hack fix untested
        #print(updates)
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
        'delta': [delta], 
        'm_bound': [len(y_bound)],
        'm_target': [len(y_target)],
        'n_classifiers': [n_classifiers],
        'seed': [seed]
    })  
    
    return results 

def draw_classifier_and_calculate_errors(w_s,sigma,n_classifiers,x_bound,y_bound,x_target,y_target,posterior_model):
    """
    As the name says, we draw classifiers and compute the necessary quantities for the different bounds
    """
    errorsum=[]
    target_errorsum=[]
    e_ssum=[]
    e_tsum=[]
    d_txsum=[]
    d_sxsum=[]
    d_tx_h=0
    d_sx_h=0
    d_tx_hprime=0
    d_sx_hprime=0
    for i in range(n_classifiers):
        
        ### do a loop and draw two classifiers for each loop
        print('Drawing classifiers...')
        h = draw_classifier(w_s, sigma, num_classifiers=1)[0]
        hprime = draw_classifier(w_s, sigma, num_classifiers=1)[0]

        """
        Calculate train and target errors
        """
        y_bound = np.array(y_bound)
        y_target = np.array(y_target)

        ######## in here we should make the results save in a vector for each part to be able to calculate
        ######## the standard deviation and be able to get error bars on things.
        print('Calculating errors, joint errors and disagreements...')
        t = time.time()
        ### for the first draw
        posterior_model.set_weights(h)
        d_tx_h=posterior_model.predict(x_target,verbose=0)
        d_sx_h=posterior_model.predict(x_bound,verbose=0)
        d_sx_h=make_01(d_sx_h)
        d_tx_h=make_01(d_tx_h)

        errorsum.append(error_from_prediction(d_sx_h,y_bound))
        target_errorsum.append(error_from_prediction(d_tx_h,y_target))
        ### for the second draw
        posterior_model.set_weights(hprime)
        d_tx_hprime=posterior_model.predict(x_target,verbose=0)
        d_sx_hprime=posterior_model.predict(x_bound,verbose=0)
        d_sx_hprime=make_01(d_sx_hprime)
        d_tx_hprime=make_01(d_tx_hprime)
        errorsum.append(error_from_prediction(d_sx_hprime,y_bound))
        target_errorsum.append(error_from_prediction(d_tx_hprime,y_target))

        e_ssum.append(joint_error(d_sx_h,d_sx_hprime,y_bound))
        d_sxsum.append(classifier_disagreement(d_sx_h,d_sx_hprime))
        e_tsum.append(joint_error(d_tx_h,d_tx_hprime,y_target))
        d_txsum.append(classifier_disagreement(d_tx_h,d_tx_hprime))
        elapsed = time.time() - t
        print('Time spent calculating errors, joint errors and disagreements: %.4fs' % elapsed)

      

        

    train_germain = np.mean(errorsum) 
    target_germain = np.mean(target_errorsum)  
    error_std = np.std(errorsum)
    target_error_std = np.std(target_errorsum)
    # Means
    e_s = np.mean(e_ssum)
    d_sx = np.mean(d_sxsum)
    e_t = np.mean(e_tsum)
    d_tx = np.mean(d_txsum)

    # Std-devs
    e_s_std = np.std(e_ssum)
    d_sx_std = np.std(d_sxsum)
    e_t_std = np.std(e_tsum)
    d_tx_std = np.std(d_txsum)
    #del posterior_model
    return e_s, e_t, d_sx, d_tx, e_s_std, e_t_std, d_sx_std, d_tx_std, train_germain, target_germain, error_std, target_error_std

    
def grid_search(train_germain,e_s,e_t,d_tx,d_sx,KL,delta,m,m_target,L,beta_bound=False,beta_inf=1):
    #### here we want to do a coarse grid search over a and omega to get the smallest bound 
    print("Starting gridsearch....")
    avec=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000]
    omegas=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000]
    tmp= sys.maxsize
    res=[]
    bestparam=[0,0]
    i = 0
    num_coeff=len(avec)*len(omegas)
    delta_p = delta/(num_coeff)
    
    for a in avec:
        for omega in omegas:
            if beta_bound:
                germain_bound, boundparts=calculate_beta_bound(e_s,d_tx,KL,delta_p,a,omega,m,m_target,L,beta_inf)
            else:
                germain_bound, a1,a2,a3,a4,a5 =calculate_germain_bound(train_germain,e_s,e_t,d_tx,d_sx,KL,delta_p,a,omega,m,L)
                boundparts=[a1,a2,a3,a4,a5]
            if min(germain_bound)<tmp:
                tmp=min(germain_bound)
                #print("Best bound thus far:"+str(tmp))
                res=germain_bound
                bestparam=[a,omega]
                bestparts=boundparts
    ### do a finer sweep around the best parameters
    if bestparam[0]!=0:
        avec=np.arange(bestparam[0]-bestparam[0]/2,bestparam[0]+bestparam[0]*4,0.1*bestparam[0])
    else:## no bound better than the max int was found, if that is even possible
        avec=np.arange(-1,1,0.1)
    if bestparam[1]!=0:
        omegas=np.arange(bestparam[1]-bestparam[1]/2,bestparam[1]+bestparam[1]*4,0.1*bestparam[1])
    else:## no bound better than the max int was found
        avec=np.arange(-1,1,0.1)
    boundparts=[0,0,0,0,0]
    delta_p = delta/(len(avec)*len(omegas) + num_coeff)
    for a in avec:
        for omega in omegas:
            if beta_bound:
                germain_bound, boundparts = calculate_beta_bound(e_s,d_tx,KL,delta_p,a,omega,m,m_target,L,beta_inf)
            else:
                germain_bound, a1,a2,a3,a4,a5 = calculate_germain_bound(train_germain,e_s,e_t,d_tx,d_sx,KL,delta_p,a,omega,m,L)
                boundparts=[a1,a2,a3,a4,a5]
                
            if min(germain_bound)<tmp:
                tmp=min(germain_bound)
                #print("Best finer bound thus far:"+str(tmp))
                res=germain_bound
                bestparts = boundparts
                bestparam=[a,omega]
                
                
    #if beta_bound==True:
      #  print("The best bound:",res)
      #  print("The best coefficients:",bestparam)            
    return res, bestparam, bestparts

def grid_search_single(train_error,KL,delta,m,MMD):
    """
    A very simple sweep over some values for beta in the mcallester mmd bound.
    """
    betas=[0.001,0.005,0.01,0.05,0.1,0.5,0.99]
    tmp= sys.maxsize
    res=[]
    bestparam=1e-9
    i = 0
    
    
    delta_p = delta/(len(betas))
    for beta in betas:
            i += 1
            bound, boundparts=calculate_mmd_bound(train_error,KL,delta_p,beta,m,MMD)
            if min(bound)<tmp:
                tmp=min(bound)
                #print("Best bound thus far:"+str(tmp))
                res=bound
                bestparam=beta
                bestparts=boundparts
    #print("The best bound:",res)
    #print("The best coefficients:",bestparam)
    return res, bestparam, bestparts
    
def calculate_mmd_bound(train_error,KL,delta,beta,m,MMD):
    L=len(KL)
    bound=[]
    a1=np.zeros(L)
    a2=np.zeros(L)
    a3=np.zeros(L)
    beta_inv=(1-beta)
    
    
    for i in range(L):
        a1[i]=train_error[i]/beta
        a2[i]=(KL[i]+np.log(1/delta))/(2*beta*beta_inv*m)
        a3[i]=MMD ## fixed for now
        bound.append(a1[i]+a2[i]+a3[i])
    boundparts=[a1,a2,a3]
    return bound, boundparts
def calculate_beta_bound(e_s,d_tx,KL,delta,b,c,m,m_target,L,BETA):
    m_s=m 
    m_t=m_target
    bprime=BETA*(b/(1-np.exp(-b)))
    cprime=c/(1-np.exp(-c))

    bound=[]
    a1=np.zeros(L)
    a2=np.zeros(L)
    a3=np.zeros(L)
    for i in range(L):
        a1[i]=cprime/2*(d_tx[i])
        a2[i]=bprime*e_s[i]
        a3[i]=(cprime/(m_t*c)+bprime/(m_s*b))*(2*KL[i]+np.log(2/delta))
    ## we cannot evaluate the eta term in the bound so this is it. For TASK 2 it is 0 anyway.
    ## And for other tasks we will not evaluate this bound anyway as we probably have no way of doing so easily..
        bound.append(a1[i]+a2[i]+a3[i])
    boundparts=[a1,a2,a3]
    return bound, boundparts



def calculate_quad_bound(KL,alpha,delta,N,train_error):
    """
    The quad bound calculated in Dziugaite et al.
    """
    N=round((1-alpha)*N)
    B=(KL+np.log(2*np.sqrt(N)/delta))/N
    ## quad bound
    bound=np.min([train_error+np.sqrt(B/2),train_error+B+np.sqrt(B*(B+2*train_error))])
    return bound