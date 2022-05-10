import os

import numpy as np
import sys
import pandas as pd
from data import mnist_m as mnistm
from data import mnist
from data import svhn
from data import usps
from data import xray
from data.label_shift import label_shift_linear, plot_splitbars, label_shift
import tensorflow.keras as keras
from data.loader import CheXpertDataGenerator

def binarize(y,x,num_labels=6):
    """
     take in one hot label encoding and make it into either 'label x' or 'not label x'
     x is in [0,5] as we have at least 6 overlapping labels in our chestxray data
     labeldict={"No Finding":0,"Cardiomegaly":1,"Edema":2,"Consolidation":3,"Atelectasis":4,"Effusion":5}
    """
    y_new=[]
    mask=np.zeros(num_labels)
    mask[x]=1
    for i in y:
        if np.dot(i,mask)==1:
            ## we have the label
            y_new.append([0,1])
        else:
            ## we do not have the label
            y_new.append([1,0])
    return np.array(y_new)
def make_mnist_binary(y):
    '''
    takes in mnist labels and returns a binarisation
    i.e. 0-4 is 0 and 5-9 is 1 for example
    '''
    
    new_y=[None]*len(y)
    for label in range(len(y)):
        if y[label][0] or y[label][1] or y[label][2] or y[label][3] or y[label][4]:
            new_y[label]=[1, 0]
        else:
            new_y[label]=[0, 1]
    return new_y
def load_task(task=2,binary=True,img_size=32,architecture='lenet'):
    """
    loads the data needed for the specific task
    """
    if task == 1 or task == 2:
        x_train, y_train, x_test, y_test = mnist.load_mnist(task)
        
        
        ###### Add train and test together 
        ### MNIST all data
        x_full=np.append(x_train,x_test, axis=0)
        y_full=np.append(y_train,y_test, axis=0)
        
        
        
        
        ## shift the distributions to create source and target distributions
        x_shift, y_shift, x_shift_target, y_shift_target = label_shift_linear(x_full,y_full,1/12,[0,1,2,3,4,5,6,7,8,9])
        
        if task==1:
            ###### label density shifted mnist
            x_source=x_shift
            y_source=y_shift
            x_target=x_shift_target
            y_target=y_shift_target
        else:
            ###########################################################################
            # MNIST + MNIST-M mix
            ###########################################################################
            x_train_m, y_train_m, x_test_m, y_test_m = mnistm.load_mnistm(y_train,y_test)
            ### MNIST-M all data
            x_full_m=np.append(x_train_m,x_test_m, axis=0)
            y_full_m=np.append(y_train_m,y_test_m, axis=0)
            ## shift the distributions to create source and target distributions
            x_shift_m, y_shift_m,x_shift_target_m, y_shift_target_m = \
        label_shift_linear(x_full_m,y_full_m,1/12,[0,1,2,3,4,5,6,7,8,9],decreasing=False)
            ##### calculate the label densities here
            densities=[]
            
            #check that this yields correct estimations of the density
            densities.append(np.sum(y_shift,axis=0))
            densities.append(np.sum(y_shift_m,axis=0))
            densities.append(np.sum(y_shift_target,axis=0))
            densities.append(np.sum(y_shift_target_m,axis=0))

            
            L=len(densities[0])
            interdomain_densities = [[] for x in range(2)]
            for i in range(L):
                ## all densities are (#samples from mnist) over (#samples from mnist-m)
                interdomain_densities[0].append(densities[0][i]/densities[1][i])
                interdomain_densities[1].append(densities[2][i]/densities[3][i])
            print(interdomain_densities)
            ## add the shifted data together to create source and target
            
            x_source=np.append(x_shift,x_shift_m, axis=0)
            y_source=np.append(y_shift,y_shift_m, axis=0)
            x_target=np.append(x_shift_target,x_shift_target_m, axis=0)
            y_target=np.append(y_shift_target,y_shift_target_m, axis=0)

   
    elif task==3:
        ###########################################################################
        # MNIST -> MNIST-M
        ###########################################################################
        
        x_train, y_train, x_test, y_test = mnist.load_mnist(task)
        x_train_m, y_train_m, x_test_m, y_test_m = mnistm.load_mnistm(y_train,y_test)
        
        ###### Add train and test together 
        ### MNIST all data
        x_full=np.append(x_train,x_test, axis=0)
        y_full=np.append(y_train,y_test, axis=0)
        
        ### MNIST-M all data
        x_full_m=np.append(x_train_m,x_test_m, axis=0)
        y_full_m=np.append(y_train_m,y_test_m, axis=0)
        
        
        x_source=x_full
        y_source=y_full
        x_target=x_full_m
        y_target=y_full_m
     
    elif task==4:
        ###########################################################################
        # MNIST -> USPS
        ###########################################################################
        x_train, y_train, x_test, y_test = mnist.load_mnist(task)
        x_train_usps, y_train_usps, x_test_usps, y_test_usps = usps.load_usps()
        
        ###### Add train and test together 
        ### MNIST all data
        x_full=np.append(x_train,x_test, axis=0)
        y_full=np.append(y_train,y_test, axis=0)
        
        ### USPS all data
        x_usps=np.append(x_train_usps,x_test_usps, axis=0)
        y_usps=np.append(y_train_usps,y_test_usps, axis=0)
        
        x_source=x_full
        y_source=y_full
        x_target=x_usps
        y_target=y_usps
    elif task==5:
        ###########################################################################
        # MNIST -> SVHN
        ###########################################################################
        
        x_train, y_train, x_test, y_test = mnist.load_mnist(task)
        x_train_svhn, y_train_svhn, x_test_svhn, y_test_svhn = svhn.load_svhn()
        
        ###### Add train and test together 
        ### MNIST all data
        x_full=np.append(x_train,x_test, axis=0)
        y_full=np.append(y_train,y_test, axis=0)
        
         ### SVHN all data
        x_svhn=np.append(x_train_svhn,x_test_svhn, axis=0)
        y_svhn=np.append(y_train_svhn,y_test_svhn, axis=0)
        
        x_source=x_full
        y_source=y_full
        x_target=x_svhn
        y_target=y_svhn
    elif task==6:
        ###########################################################################
        # chexpert + chestxray14 mix
        ###########################################################################
        from sklearn.model_selection import train_test_split
        #data_path2="/cephyr/NOBACKUP/groups/snic2021-23-538/"
        data_path="/cephyr/users/adambre/Alvis/mnist_transfer/"
        x_chest=np.load(data_path+"chestxray14_"+str(img_size)+".npy",allow_pickle=True)
        y_chest=np.load(data_path+"chestxray14_"+str(img_size)+"_labels.npy",allow_pickle=True)

        x_chex=np.load(data_path+"chexpert_"+str(img_size)+".npy",allow_pickle=True)
        y_chex=np.load(data_path+"chexpert_"+str(img_size)+"_labels.npy",allow_pickle=True)

        


        ### do standard scaling
        x_chest = x_chest.astype('float32')
        sigma=np.std(x_chest)
        x_chest /=sigma

        x_chex = x_chex.astype('float32')
        sigma2=np.std(x_chex)
        x_chex /=sigma
        ## mean subtraction
        mu=np.mean(x_chest)
        x_chest -= mu

        mu2=np.mean(x_chex)
        x_chex -= mu
        ## print amount of each labels in the dataset
        #print(np.sum(y_chex,axis=0))
        #print(np.sum(y_chest,axis=0))
        #         print(y_chest[0])
        #         print(y_chex[53])
        #         sys.exit(-1)
        
        
        ## two datasets of different lengths, want to induce domain imbalance between source and target, 
        ## we do 20% of samples from each label in chestxray14 is added to chexpert to create the source, rest is target, i.e. 
        ## there is no chexpert in the target AND source is much larger than the target
        x_source=x_chex
        y_source=y_chex
        
        x_sourceadd=[]
        y_sourceadd=[]
        x_target=[]
        y_target=[]
        for label in range(5):
            # remove some percentage of each label, now 20% -> beta_inf=4
            x_shift, y_shift, x_shift_target, y_shift_target = label_shift(x_chest,y_chest,0.2,label)
            x_chest=x_shift
            y_chest=y_shift
            # append to source
            x_sourceadd.extend(x_shift_target)
            y_sourceadd.extend(y_shift_target)
            
        # target is the remaining samples from chestxray14
        
  
        x_source=list(x_source)
        y_source=list(y_source)
        x_source.extend(x_sourceadd)
        y_source.extend(y_sourceadd)
        x_source=np.array(x_source)
        y_source=np.array(y_source)
    

        x_target=x_chest
        y_target=y_chest

        ### Binarize labels

        y_source=binarize(y_source,2)
  
        y_target=binarize(y_target,2)
        
        print("source",len(y_source))
        print(np.sum(y_source,axis=0))
        print("-"*40)
        print("target",len(y_target))
        print(np.sum(y_target,axis=0))
        
      

    elif task==7:
        ###########################################################################
        # chexpert + chestxray14 with dataloader
        ###########################################################################
        from sklearn.model_selection import train_test_split
        batch_size=64
        #data_path="/home/users/adam/Code/Datasets/"
        data_path="/cephyr/NOBACKUP/groups/snic2021-23-538/"
        chest_image_source_dir=data_path+"chestXray14/"
        chex_image_source_dir=data_path+"new_chexpert/"
        
#         output_dir="/home/adam/Code"
#         class_names= ["No Finding","Cardiomegaly","Edema","Consolidation","Atelectasis","Effusion"]
        
        ### load and print csv files
        
        
        
        
        # chexpert csv files
#         train_file = os.path.join(chex_image_source_dir, 'train.csv')
#         valid_file = os.path.join(chex_image_source_dir, 'valid.csv')
#         chest_file = os.path.join(chest_image_source_dir, 'Data_Entry_2017_v2020.csv')
#         data = pd.read_csv(train_file)
#         data_v = pd.read_csv(train_file)
#         data2=pd.read_csv(chest_file)
#         print(data2)
#         for label in data2["Finding Labels"]:
            
    
#             if isinstance(label,float):
#                 print(label)
#                 print(label.index)
#         sys.exit(-1)
        
#         print(data.head())
#         print(data2.head())
        
        #### fix the labels and such to be on desired form
        chest_data, _ =xray.make_xray14_labels(chest_image_source_dir)
        chest_data=chest_data.sort_values(by=["Patient_ID","Follow_Up_#"])
        #print("-"*40)
        #print(chest_data.head())
        
        
        chex_data, _=xray.make_chexpert_labels(chex_image_source_dir)
        #print("-"*40)
        #print(chex_data.head())
        
        #### do split to source and target 
        y_chex = chex_data["Finding_Labels"]
        y_chest = chest_data["Finding_Labels"]
        x_chex = chex_data#.drop("Finding_Labels", axis=1)
        x_chest = chest_data#.drop("Finding_Labels", axis=1)
        
        x_shift, x_shift_target,y_shift, y_shift_target = train_test_split(x_chest,y_chest,test_size=0.2)
#         print("--"*30)
#         print(x_shift_target)
#         print(x_chest)
#         print("--"*30)
#         sys.exit(-1)
        # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=random_state)
        y_source=pd.concat([y_shift_target,y_chex])
        x_source=pd.concat([x_shift_target,x_chex])
        y_target=y_shift
        x_target=x_shift
        
        
        #### return dataloader
        source_data_loader= CheXpertDataGenerator(dataset_df=x_source, y=y_source,
            batch_size=batch_size,
            target_size=(img_size, img_size),
                   )
        
        target_data_loader= CheXpertDataGenerator(dataset_df=x_target,y=y_target,
            batch_size=batch_size,
            target_size=(img_size, img_size),
                    )
    else: 
        raise Exception('Task '+str(task)+' does not exist')
     
    #### binarize if chosen
    if binary:
        if task==6 or task==7:
            pass # we have already done this
        else:
            y_source=make_mnist_binary(y_source)
            y_target=make_mnist_binary(y_target)

    

    if task==7:
        return source_data_loader,target_data_loader
    else:
               #####################################################################################################################
        # Shuffle the data with some seed before returning it. 
        # This is done to mitigate data being in too specific an order, i.e. mixes being one dataset after the other etc.
        #####################################################################################################################
        np.random.seed()
        L=len(y_source)
        source_indices=np.random.permutation(L)
        L2=len(y_target)
        target_indices=np.random.permutation(L2)
        x_source=x_source[source_indices]

        x_target=x_target[target_indices]
        return x_source, np.array(y_source)[source_indices], x_target, np.array(y_target)[target_indices]
