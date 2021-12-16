import tensorflow as tf
import numpy as np
import os
from experiments.models import *
import gc, re, copy
import pandas as pd
import pickle
## set the project folder to something for saving
project_folder="/home/adam/code/"
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

            #print("\n Saved weights at the start of batch"+str(batch)+"\n")

            ## Create folder
            weight_path = self.filepath+"/1_"+str(batch)+".ckpt"
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            
            self.model.save_weights(weight_path)
            
def train_posterior(alpha,x_train,y_train,prior_weights=None,x_test=[],y_test=[],save=True,epsilon=0.01,Task=2,Binary=False,batch_size=128,architecture="lenet"):
        
        TASK=Task
        
        ### x_test should be the whole of S for early stopping purposes
    
        checkpoint_path = "posteriors/"+"task"+str(TASK)+"/"+str(architecture)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
        if Binary:
            checkpoint_path = "posteriors/"+"task"+str(TASK)+"/Binary/"+str(architecture)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
        
        
        # Create a callback that saves the model's weights every epoch
        checkpoint_freq=np.ceil(len(x_train)/batch_size)
        #print(checkpoint_freq)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            save_freq="epoch",#int(checkpoint_freq),### 547 = ceiling(70000/128) i.e training set for MNIST/MNIST-M,
            filepath=checkpoint_path+"/2_{epoch:0d}.ckpt", 
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
                ## tune when to save as needed for plots
        )
        ## callback for first part
        fast_checkpoint_freq=np.ceil(len(x_train)/(batch_size*10))
        fast_cp_callback =fast_checkpoints(checkpoint_path,int(fast_checkpoint_freq))
        stopping_callback=stop_callback(monitor='val_acc',value=1-epsilon)
    
        M=init_task_model(TASK,Binary,architecture)

        
            
        ## choose loss function, optimiser etc. and train
        
        M.compile(loss=tf.keras.losses.categorical_crossentropy,
               optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
        
        
        ## Create the folder
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        ## remove previous weights
        import glob
        files=glob.glob(os.path.join(checkpoint_path+'/*'))
        
        
        for file in files:
            if file==(checkpoint_path+'/params.txt'):
                files.remove(checkpoint_path+'/params.txt')
            os.remove(file)
        
        ### load the prior weights
        if prior_weights is not None:
            M.set_weights(prior_weights)
        elif(alpha==0):
            ### save the rand. init as the prior
            if Binary:
                prior_path="priors/"+"task"+str(TASK)+"/Binary/"+str(architecture)+"/"+str(int(100*alpha))+"/prior.ckpt"
            else:
                prior_path="priors/"+"task"+str(TASK)+"/"+str(architecture)+"/"+str(int(100*alpha))+"/prior.ckpt"
            
            ## Create the folder
            os.makedirs(os.path.dirname(prior_path), exist_ok=True)

            ## Save the weights
            M.save_weights(prior_path)
        else:
            if Binary:
                prior_path="priors/"+"task"+str(TASK)+"/Binary/"+str(architecture)+"/"+str(int(100*alpha))+"/prior.ckpt"
                M.load_weights(prior_path).expect_partial()
            else:
                
                M.load_weights(prior_path).expect_partial()
        
    
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
           verbose=0,
                        )
        
        print("-"*40)
        print("Finished training first posterior epoch")
        if save:
            CALLBACK=[cp_callback,stopping_callback]
        else:
            CALLBACK=[stopping_callback]
            
        fit_info = M.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=2000, # we should have done early stopping before this completes
           callbacks=CALLBACK,
           validation_data=(x_test, y_test),
           verbose=0,
                        )
        print("-"*40)
        print("Finished training posterior")
        
         #### save the last posterior weights to disk
        #epochs_trained=len(fit_info.history['loss'])
        #if save:
            ## Create the folder
            #os.makedirs(os.path.dirname(checkpoint_path+"/2_"+str(epochs_trained)), exist_ok=True)
            
            
            #M.save_weights(checkpoint_path+"/2_"+str(epochs_trained)) ###### check if we need this; TODO!!!!!!
            
        #### save textfile with parameters, i.e. alpha ,epochs trained and epsilon
#         if Binary:
#             with open('posteriors/'+"task"+str(TASK)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+'/params.txt', 'w') as f:
#                 f.write('\n'.join([str(alpha), str(epsilon), str(epochs_trained)]))     
#             f.close()
#         else:
#             with open('posteriors/'+"task"+str(TASK)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+'/params.txt', 'w') as f:
#                 f.write('\n'.join([str(alpha), str(epsilon), str(epochs_trained)]))     
#             f.close()
        
        return M.get_weights()
    
def train_prior(alpha,total_epochs,x_train=[],y_train=[],x_target=[],y_target=[],save=True,Task=2,Binary=False,batch_size=128,architecture="lenet"):
    TASK=Task
    
    if Binary:
        ## Create the folders
        os.makedirs(os.path.dirname("priors/"+"task"+str(TASK)+"/Binary/"+str(architecture)+"/"+str(int(100*alpha)))+"/", exist_ok=True)
        checkpoint_path = "priors/"+"task"+str(TASK)+"/Binary/"+str(architecture)+"/"+str(int(100*alpha))
    else:
        os.makedirs(os.path.dirname("priors/"+"task"+str(TASK)+"/"+str(architecture)+"/"+str(int(100*alpha)))+"/", exist_ok=True)
        checkpoint_path = "priors/"+"task"+str(TASK)+"/"+str(architecture)+"/"+str(int(100*alpha))
    ## remove previous weights
    import glob
    files=glob.glob(os.path.join(checkpoint_path+'/*'))
    for file in files:
        os.remove(file)
    M=init_task_model(TASK,Binary,architecture)
    #sys.exit(-1)
    ### save checkpoints 10 times over the training of the prior
    l=len(x_train)
    checkpoint_freq=np.ceil(l/(batch_size*10))
    fast_cp_callback =fast_checkpoints(checkpoint_path,int(checkpoint_freq))
    if save:
            CALLBACK=[fast_cp_callback]
    else:
            CALLBACK=[]
            
    ## choose loss function, optimiser etc. and train
    M.compile(loss=tf.keras.losses.categorical_crossentropy,
               optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                      metrics=['accuracy'],)
    fit_info = M.fit(x_train, y_train,
           batch_size=batch_size,
           callbacks=CALLBACK,
           epochs=total_epochs,
           verbose=0,
                        )
    print("-"*40)
    print("Finished training prior")
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

        model=init_task_model(TASK,Binary)
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
               optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.95),
                  metrics=['accuracy'],)
      
            
        model.load_weights(checkpoint_path+"/"+str(checkpoint)+".ckpt").expect_partial()
        target_error.append(1-model.evaluate(x_target,y_target,verbose=0)[1])
        error.append(1-model.evaluate(x_train,y_train,verbose=0)[1])
    print("-"*40)
    print("Finished computing prior sample and target errors")
    if save:
        results=pd.DataFrame({'Weightupdates': weight_updates,
            'Trainerror': error,
            'targeterror':target_error,
            })
       
        ## Create the folders
        os.makedirs(os.path.dirname(project_folder+"mnist_transfer/results/"+"task"+str(TASK)+"/Binary/"+str(architecture)+"/"), exist_ok=True)
        if Binary:
            with open(project_folder+"mnist_transfer/results/"+"task"+str(TASK)+"/Binary/"+str(architecture)+"/"+str(int(100*alpha))+"_prior_results.pkl",'wb') as f:
                pickle.dump(results,f)
            f.close()
        else:
            with open(project_folder+"mnist_transfer/results/"+"task"+str(TASK)+"/"+str(architecture)+"/"+str(int(100*alpha))+"_prior_results.pkl",'wb') as f:
                pickle.dump(results,f)
            f.close()
    return model.get_weights()
    
