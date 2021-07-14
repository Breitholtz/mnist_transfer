'''
def train_prior_and_posterior(alpha,x_train,y_train,x_test,y_test,b=256,sigma=1e-3,epsilon=0.1):
    
    
    """
    alpha is the proportion of the training data used for the prior 
    
    b is the batch size, 256 in Dziugaite
    
    """
    from sklearn.model_selection import train_test_split
    
    x_bound, x_prior, y_bound , y_prior = train_test_split(x_train,y_train,test_size=alpha)
    
    N=len(x_prior)

    
    w_a=train_prior(alpha,total_epochs=round(N/b),x_train=x_prior,y_train=y_prior)
    
    #### train posterior until some error/loss has been achieved, then terminate
    w_s, train_acc=train_posterior(alpha,x_train,y_train,w_a,x_test,y_test,epsilon=epsilon)
    
  
    
    ## should we return anything?
    return w_a,w_s,N,train_acc
'''

import numpy as np
import os
import tensorflow as tf
from importlib.machinery import SourceFileLoader
path_to_root_file="/home/adam/Code/"
module5 = SourceFileLoader('train', path_to_root_file+'mnist_transfer/experiments/training.py').load_module()
from train import *

## custom callback to terminate training at some specific value of a metric
class stop_callback(tf.keras.callbacks.Callback):
    def __init__(self, monitor='accuracy', value=0.001, verbose=0):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy')> self.value): # select the accuracy
            print("\n !!! training error threshold reached, no further training !!!")
            self.model.stop_training = True
            


def calculate_bound(KL,alpha,delta,N,train_error):
    N=round((1-alpha)*N)
    B=(KL+np.log(2*np.sqrt(N)/delta))/N
    ## quad bound
    bound=np.min([train_error+np.sqrt(B/2),train_error+B+np.sqrt(B*(B+2*train_error))])
    return bound
def draw_classifier(weights,sigma,num_classifiers):
        
        w_s_draw=[[]]*num_classifiers
        #print(np.array(w_s_draw[0]))
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
'''
def create_prior():
    for i in range(2,9):
        sigmas.append(3*10**(-i))
        if(i==8):
            break
        sigmas.append(10**(-i))

    global TASK
    prior_path = "priors/"+"task"+str(TASK)+"/Binary/"+str(int(0.1*100))+"/prior.ckpt"


    model=init_MNIST_model_binary()
            ## choose loss function, optimiser etc. and train
    model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                          metrics=['accuracy'],)
    model.load_weights(prior_path).expect_partial()
    w_a=model.get_weights()
    w_prior=draw_classifier(w_a,sigma=1,iterations=1,no_add=True)
    w_prior=np.mean(w_prior,axis=0)
    model.set_weights(w_prior)
        #### save the prior weights to disk
    checkpoint_path = "priors/"+"task"+str(TASK)+"/Binary/"+str(int(1*100))+"/prior.ckpt"
    model.save_weights(checkpoint_path)
'''