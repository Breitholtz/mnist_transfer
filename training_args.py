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
from util.kl import *
from util.misc import *
from results.plotting import *

# Hyper-parameters
batch_size = 128
num_classes = 10
epochs = 10
make_plots = False

delta=0.05 ## what would this be?   
# def unpack(model, training_config, weights):
#     restored_model = deserialize(model)
#     if training_config is not None:
#         restored_model.compile(
#             **saving_utils.compile_args_from_training_config(
#                 training_config
#             )
#         )
#     restored_model.set_weights(weights)
#     return restored_model

# # Hotfix function
# def make_keras_picklable():
#     def __reduce__(self):
#         model_metadata = saving_utils.model_metadata(self)
#         training_config = model_metadata.get("training_config", None)
#         model = serialize(self)
#         weights = self.get_weights()
#         return (unpack, (model, training_config, weights))

#     cls = Model
#     cls.__reduce__ = __reduce__
TASK=2 ### hack
def read_weights(model,w_a,x_bound,y_bound,x_target,y_target,sigma,epsilon,alpha,Binary=False,Task=TASK):
    batch_size=128
    batches_per_epoch=np.ceil(len(y_target)/batch_size) ## should be 547
    epoch=1
    
    # Run the function to fix pickling issue
    make_keras_picklable()
    
    
    sigma=sigma[0]*10**(-1*sigma[1])    
    
    ### Here we do something more intelligent to not have to hardcode the epoch amounts. 
    ### we parse the filenames and sort them in numerical order and then load the weights
    if Binary:
        path="posteriors/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
    else:
        path="posteriors/"+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
        #epochs_trained
    #epochs = [] #list of 
    list1=[]
    list2=[]
    dirFiles = os.listdir(path) #list of directory files
    ## remove the ckpt.index and sort so that we get the epochs that are in the directory
    for files in dirFiles: #filter out all non checkpoints
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
    
    weight_updates=[]
    for i in Ws:
        #print(i)
        if i[0]=="1":
            if i[1]=="_":
                weight_updates.append(int(i[2:]))
    for i in list2:
        weight_updates.append((int(i[2:])+1)*batches_per_epoch)
   
    ### load the model and the weights
    N_checkpoints=len(Ws)
    KLs=np.zeros(N_checkpoints)
    errors=np.zeros(N_checkpoints)
    targeterrors=np.zeros(N_checkpoints)
    epochs=np.zeros(N_checkpoints)
#     for checkpoint in Ws:
    
    #### here we should pass all the checkpoints to different processes and evaluate on the dataset
    args=[]
    #for i in range(N_checkpoints):
    #for i in range(2):
        #args.append(nnp.array(i,Ws[i],path,KLs, errors,targeterrors,epochs,model,w_a,copy.deepcopy(x_bound),copy.deepcopy(y_bound),copy.deepcopy(x_target),copy.deepcopy(y_target),sigma,epsilon,alpha,Binary,TASK))
        #args.append(np.array([i,Ws[i],path,KLs, errors,targeterrors,epochs,model,w_a,x_bound,y_bound,x_target,y_target,sigma,epsilon,alpha,Binary,TASK]))
    args.append([Ws[0],x_bound,y_bound])   
    args.append([Ws[1],copy.deepcopy(x_bound),copy.deepcopy(y_bound)])
    args.append([Ws[2],copy.deepcopy(x_bound),copy.deepcopy(y_bound)])   
    args.append([Ws[3],copy.deepcopy(x_bound),copy.deepcopy(y_bound)])
    p = Pool(processes = 3)
    print("could deep copy")
    #for arg in args:
    if __name__ == '__main__':
        p.imap(dumb_func,args)
        p.close()
        p.join()

    #print("we made it here!!!!")
    print(KLs)
    print(errors)
    print(targeterrors)
    sys.exit(-1)
    
    return KLs,errors,targeterrors,Ws,Xvector

def dumb_func(args):
    #print(args)
    init_tf()
    
    
    print("!!!!")
    model=init_MNIST_model_binary()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),metrics=['accuracy'])
    print("Model compiled")
    path="posteriors/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
    model.load_weights(path+"/"+str(checkpoint_name)+".ckpt").expect_partial()
    print(w_s=model.get_weights())
    #model.evaluate(args[1],args[2])
    sum=0
    for i in range(1000000):
        sum+=i
    print(sum)
    print(len(args[0]))
    
def init_tf():
    ### making sure that we have the GPU to work on
    gpus = tf.config.experimental.list_physical_devices('GPU')
    #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
      # I do not know why I have to do this but gpu does not work otherwise.
        try:
            tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
            print(e)

    
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
            
class fast_checkpoints(tf.keras.callbacks.Callback):
    def __init__(self,checkpoint_path,save_freq):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.save_freq=save_freq
        self.filepath=checkpoint_path
        self.verbose=1
        self.save_best_only=False
        self.save_weights_only=True
    def on_train_batch_begin(self, batch, epoch, logs=None):
         if batch%self.save_freq==0:

            print("\n Saved weights at the start of batch"+str(batch)+"\n")

            ## Create folder
            weight_path = self.filepath+"/1_"+str(batch)+".ckpt"
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            
            self.model.save_weights(weight_path)
            
def train_posterior(alpha,x_train,y_train,prior_weights=None,x_test=[],y_test=[],save=True,epsilon=0.01,Task=2,Binary=False):
        
        TASK=Task
        batch_size=128
        
        ### x_test should be the whole of S for early stopping purposes
    
        checkpoint_path = "posteriors/"+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
        if Binary:
            checkpoint_path = "posteriors/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
        
        
        # Create a callback that saves the model's weights every epoch
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            save_freq=547,   ### 547 = ceiling(70000/128) i.e training set for MNIST/MNIST-M,
            filepath=checkpoint_path+"/2_{epoch:0d}.ckpt", 
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
                ## tune when to save as needed for plots
        )
        fast_cp_callback =fast_checkpoints(checkpoint_path,45)
        stopping_callback=stop_callback(monitor='val_acc',value=1-epsilon)
    
        if Binary:
            M=init_MNIST_model_binary()
        else:
            M=init_MNIST_model()

        
            
        ## choose loss function, optimiser etc. and train
        
        M.compile(loss=tf.keras.losses.categorical_crossentropy,
               optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
        ### load the prior weights
        if prior_weights is not None:
            M.set_weights(prior_weights)
        elif(alpha==0):
            ### save the rand. init as the prior
            prior_path="priors/"+"task"+str(TASK)+"/Binary/"+str(int(100*alpha))+"/prior.ckpt"
            
            ## Create the folder
            os.makedirs(os.path.dirname(prior_path), exist_ok=True)
            
            ## Save the weights
            M.save_weights(prior_path)
        else:
            if Binary:
                prior_path="priors/"+"task"+str(TASK)+"/Binary/"+str(int(100*alpha))+"/prior.ckpt"
                M.load_weights(prior_path)#.expect_partial()
            else:
                prior_path="priors/"+"task"+str(TASK)+"/"+str(int(100*alpha))+"/prior.ckpt"
                M.load_weights(prior_path)#.expect_partial()
        
    
        if save:
            CALLBACK=[fast_cp_callback,stopping_callback]
        else:
            CALLBACK=[stopping_callback]
        ### train for one epoch with more checkpoints to be able to plot more there
        fit_info = M.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=1, 
           callbacks=CALLBACK,
           validation_data=(x_test, y_test),
           verbose=1,
                        )
        
        
        if save:
            CALLBACK=[cp_callback,stopping_callback]
        else:
            CALLBACK=[stopping_callback]
            
        fit_info = M.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=2000, # we should have done early stopping before this completes
           callbacks=CALLBACK,
           validation_data=(x_test, y_test),
           verbose=1,
                        )
        
        
         #### save the last posterior weights to disk
        epochs_trained=len(fit_info.history['loss'])
        if save:
            ## Create the folder
            os.makedirs(os.path.dirname(checkpoint_path+"/2_"+str(epochs_trained)), exist_ok=True)
            
            M.save_weights(checkpoint_path+"/2_"+str(epochs_trained)) ###### check if we need this; TODO!!!!!!
            
        #### save textfile with parameters, i.e. alpha ,epochs trained and epsilon
        if Binary:
            with open('posteriors/'+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+'/params.txt', 'w') as f:
                f.write('\n'.join([str(alpha), str(epsilon), str(epochs_trained)]))     
            f.close()
        else:
            with open('posteriors/'+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+'/params.txt', 'w') as f:
                f.write('\n'.join([str(alpha), str(epsilon), str(epochs_trained)]))     
            f.close()
        W=M.get_weights()
        return W
    
def train_prior(alpha,total_epochs,x_train=[],y_train=[],x_target=[],y_target=[],save=True,Task=2,Binary=False):
    TASK=Task
    checkpoint_path = "priors/"+"task"+str(TASK)+"/"+str(int(100*alpha))
    
    if Binary:
        M=init_MNIST_model_binary()
        checkpoint_path = "priors/"+"task"+str(TASK)+"/Binary/"+str(int(100*alpha))#+"/prior.ckpt"
    else:
        M=init_MNIST_model()
        
    fast_cp_callback =fast_checkpoints(checkpoint_path,10)
    if save:
            CALLBACK=[fast_cp_callback]
    else:
            CALLBACK=[]
            
    ## choose loss function, optimiser etc. and train
    M.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
    fit_info = M.fit(x_train, y_train,
           batch_size=batch_size,
           callbacks=CALLBACK,
           epochs=total_epochs,
           verbose=1,
                        )
    #### save the final prior weights to disk
    if save:
        os.makedirs(checkpoint_path, exist_ok=True)
        M.save_weights(checkpoint_path+"/prior.ckpt")
     
    
    list1=[]
    
    dirFiles = os.listdir(checkpoint_path) #list of directory files
    
    ## remove the ckpt.index and sort so that we get the epochs that are in the directory
    for files in dirFiles: #filter out all non weights
        if '.ckpt.index' in files:
            name = re.sub('\.ckpt.index$', '', files)
            if (name[0]=="1"):
                list1.append(name)
        
    list1.sort(key=lambda f: int(re.sub('\D', '', f)))
    list1.append("prior")    ## add the final weights which has no number
    
    Ws=list1
    weight_updates=[]
    for i in Ws:
        if i[0]=="1":
            if i[1]=="_":
                weight_updates.append(int(i[2:]))
    weight_updates.append(int(np.ceil(len(y_train)/batch_size)))
    
    error=[]
    target_error=[]
    for checkpoint in Ws:
        if Binary:
            model=init_MNIST_model_binary()
            model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
        else:
            model=init_MNIST_model()
            model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
            
        model.load_weights(checkpoint_path+"/"+str(checkpoint)+".ckpt")#.expect_partial()
        target_error.append(1-model.evaluate(x_target,y_target,verbose=0)[1])
        error.append(1-model.evaluate(x_train,y_train,verbose=0)[1])
    
    if save:
        results=pd.DataFrame({'Weightupdates': weight_updates,
            'Trainerror': error,
            'targeterror':target_error,
            })
        with open(path_to_root_file+'mnist_transfer/'+checkpoint_path+"/results.pkl",'wb') as f:
            pickle.dump(results,f)
        f.close()
    
    return model.get_weights()
    
def read_and_prepare_results(alpha,x_bound,y_bound,x_target,y_target,sigma,delta,N,epsilon,Binary=False,Task=TASK):
    
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
    
    # initialise model
    if Binary:
        M=init_MNIST_model_binary()
    else:
        M=init_MNIST_model()
    M.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
    ### load the prior weights if there are any
    if alpha==0:
        ### do nothing, i.e take the random init
        w_a=M.get_weights()
    else:
        M.load_weights(prior_path).expect_partial()
        w_a=M.get_weights()
    print(Binary)
    # read the weights and calculate what is needed for the bound
    [KLs,errors,targeterrors,Ws,weight_updates]=read_weights(M,w_a,x_bound,y_bound,x_target,y_target,sigma_tmp,epsilon,alpha,Binary=Binary,Task=TASK)    
    
    #print(KLs)
    #print(errors)
    #print(targeterrors)
    #print(Ws)
   
    bound=[]
    ### calculate the bound
    for i in range(len(weight_updates)):
        bound.append(calculate_bound(KLs[i],alpha,delta,N,errors[i]))
    
    
    #save the results to a pickled dataframe in results
    results=pd.DataFrame({'Weightupdates': weight_updates,
        'Trainerror': errors,
        'targeterror':targeterrors,
        'KL': KLs,
        'Bound': bound})
    with open(path_to_root_file+'mnist_transfer/'+result_path+str(sigma_tmp[0])+str(sigma_tmp[1])+"_results.pkl",'wb') as f:#int(sigma*10**8)
        pickle.dump(results,f)
    f.close()
    return results

def plot_result_file(epsilon,alpha,sigma,Binary=False,Task=TASK):
    sigma_tmp=sigma
    sigma=sigma[0]*10**(-1*sigma[1])
    
    if Binary:
        result_path="results/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"+str(sigma_tmp[0])+str(sigma_tmp[1])
        plt.title("Binary: "+r"$\alpha$="+str(alpha)+r" $\epsilon$="+str(epsilon)+r" $\sigma$="+str(sigma))
    else:
        result_path="results/"+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"+str(sigma_tmp[0])+str(sigma_tmp[1])
        plt.title(r"$\alpha$="+str(alpha)+r" $\epsilon$="+str(epsilon)+r" $\sigma$="+str(sigma))
    results=pd.read_pickle(result_path+"_results.pkl")
    
    ### do the plots
    plt.plot(results["Weightupdates"],results["Bound"],'r*-')
    plt.plot(results["Weightupdates"],results["Trainerror"],'m^-')
    
    plt.xlabel("Weight updates")
    plt.ylabel("Error")
    
    plt.legend(["Bound","Empirical error"])
    plt.show()

def find_optimal_sigma(sigmas,epsilon, alpha,Binary=False,Task=TASK):
    #### to find the optimal sigma just do a search through all the results 
    #### and save the one for each parameter which has the minimal bound
    #### Do we do this per epoch or for some other value? The sigma which yields the lowest bound overall for some epoch?
    optimal=[0,1]
    # search through all epochs and pick the sigma which yields the smallest bound during the whole training process
    for sigma in sigmas:
        sigma_tmp=sigma
        sigma=sigma[0]*10**(-1*sigma[1])
        if Binary:
            result_path="results/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"+str(sigma_tmp[0])+str(sigma_tmp[1])
        else:
            result_path="results/"+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"+str(sigma_tmp[0])+str(sigma_tmp[1])
        results=pd.read_pickle(result_path+"_results.pkl")
        MIN=np.min(results["Bound"])
        if (MIN<optimal[1]):
            optimal[1]=MIN
            optimal[0]=sigma
    print("The optimal sigma is {} with bound value {}".format(optimal[0],optimal[1]))

   
#### find the optimal sigma for every combination of parameters

#### use the optimal sigmas to calculate the bound 50 times(with different data orders and initialisation)
#### (Note: also delta=13*delta_0) for every combination and save the mean and std for plotting into a result file
#for i in range(50):
    ## take in the data and split with a new seed
 #   x_bound, x_prior, y_bound , y_prior = train_test_split(x_source,y_source,test_size=alpha,random_state=(69105+i))
#### 
def read_prior(alpha,TASK=2,Binary=True):
    checkpoint_path = "priors/"+"task"+str(TASK)+"/"+str(int(100*alpha))
    if Binary:
        checkpoint_path = "priors/"+"task"+str(TASK)+"/Binary/"+str(int(100*alpha))
    result_path=path_to_root_file+'mnist_transfer/'+checkpoint_path+"/results.pkl"
    results=pd.read_pickle(result_path)
    plt.title(r"$\alpha$="+str(alpha))
    plt.plot(results["Weightupdates"],results["Trainerror"],'m^-')
    plt.plot(results["Weightupdates"],results["targeterror"],'k^-')
    plt.legend(["Training error","Target error"])
    
def plot_prior_and_posterior(alpha,epsilon,sigma,TASK=2,Binary=True):
    ### load in the prior data
    sigma_tmp=sigma
    sigma=sigma[0]*10**(-1*sigma[1])
    checkpoint_path = "priors/"+"task"+str(TASK)+"/"+str(int(100*alpha))
    if Binary:
        checkpoint_path = "priors/"+"task"+str(TASK)+"/Binary/"+str(int(100*alpha))
    result_path=path_to_root_file+'mnist_transfer/'+checkpoint_path+"/results.pkl"
    results=pd.read_pickle(result_path)
    result_path_post="results/"+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"+str(sigma_tmp[0])+str(sigma_tmp[1])
    results2=pd.read_pickle(result_path_post+"_results.pkl")
    
    
    ### remove/ignore the last entry of the prior data 
    ### as it should be a duplication of the first one from the posterior results
    
    ### training error
    A=list(results2["Weightupdates"]+list(results["Weightupdates"])[-1])
    B=list(results["Weightupdates"])[:-1]
    B.extend(A)
    C=list(results["Trainerror"])[:-1]
    C.extend(list(results2["train_germain"]))
    plt.plot(B,C,'-m^')
    
    ## target error
    D=list(results["targeterror"])[:-1]
    D.extend(list(results2["target_germain"]))
    plt.plot(B,D,'-k*')
    
    ### bound
    E=results2["germain_bound"]
    plt.plot(A,E,'-D')
    F=results2['boundpart3_germain']
    plt.plot(A,F,'-o')
    print(results2["target_germain"])
    print(results2["germain_bound"])
    ### lines for uninformative region and worse than random guessing; also for end of prior training
    plt.axvline(A[0],color="grey")
    plt.axhline(y=0.5, color="black", linestyle="--")
    plt.axhline(y=1, color="red", linestyle="--")
    plt.legend(["Training error","Target error","Bound","KL-part"])

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
        #""" Model definition """
    
        
    """ Model training/computing bounds """
    y_source_bin=np.array(make_mnist_binary(y_source))
    y_target_bin=np.array(make_mnist_binary(y_target))
    
    print("Alpha is:"+str(alpha))
    if alpha!=0:
        x_bound, x_prior, y_bound , y_prior = train_test_split(x_source,y_source_bin,test_size=alpha,random_state=69105)
        #w_a=train_prior(alpha,1,x_source,y_source_bin,x_target=x_target,y_target=y_target_bin,save=True,Task=2,Binary=True)
    w_s=train_posterior(alpha,x_source,y_source_bin,None,x_test=x_source,y_test=y_source_bin,
                             epsilon=epsilon,Task=task,Binary=True)
                
    print(seed)
    print(epsilon)
    print(sigma)
    print(task)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train simple MNIST model.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, dest='batch_size')
    parser.add_argument('-r', '--seed', type=int, default=0, dest='seed')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, dest='alpha')
    parser.add_argument('-e', '--epsilon', type=float, default=0.01, dest='epsilon')
    parser.add_argument('-s', '--sigma', nargs='+', type=int , default="3 2", dest='sigma')
    parser.add_argument('-t', '--task', type=int, default=2, dest='task')


    args = parser.parse_args()

    main(batch_size=int(args.batch_size), seed=int(args.seed), alpha=float(args.alpha),epsilon=float(args.epsilon), sigma=args.sigma,task=int(args.task))
