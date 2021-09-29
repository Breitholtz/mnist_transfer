import numpy as np

from data import mnist_m as mnistm
from data import mnist
from data import svhn
from data import usps
from data.label_shift import label_shift_linear

def load_task(TASK=2):
    ### loads the data needed for the specific task

    if TASK == 1 or TASK == 2:
        x_train, y_train, x_test, y_test = mnist.load_mnist()
        x_train_m, y_train_m, x_test_m, y_test_m = mnistm.load_mnistm(y_train,y_test)
        
        ###### Add train and test together 
        ### MNIST all data
        x_full=np.append(x_train,x_test, axis=0)
        y_full=np.append(y_train,y_test, axis=0)
        
        ### MNIST-M all data
        x_full_m=np.append(x_train_m,x_test_m, axis=0)
        y_full_m=np.append(y_train_m,y_test_m, axis=0)
        
        
        ## shift the distributions to create source and target distributions
        x_shift, y_shift, x_shift_target, y_shift_target = label_shift_linear(x_full,y_full,1/12,[0,1,2,3,4,5,6,7,8,9])
        x_shift_m, y_shift_m,x_shift_target_m, y_shift_target_m = \
        label_shift_linear(x_full_m,y_full_m,1/12,[0,1,2,3,4,5,6,7,8,9],decreasing=False)
        if TASK==1:
            ###### label density shifted mnist
            x_source=x_shift
            y_source=y_shift
            x_target=x_shift_target
            y_target=y_shift_target
        else:
            #### MIXED MNIST and MNIST-m
            
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
            ## add the shifted data together to create source and target
            x_source=np.append(x_shift,x_shift_m, axis=0)
            y_source=np.append(y_shift,y_shift_m, axis=0)
            x_target=np.append(x_shift_target,x_shift_target_m, axis=0)
            y_target=np.append(y_shift_target,y_shift_target_m, axis=0)
       
    elif TASK==3:
        #### MNIST -> MNIST-m
        ### load MNIST and MNIST-M
        x_train, y_train, x_test, y_test = mnist.load_mnist()
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
     
    elif TASK==4:
        #raise Exception('Not implemented/tested')
        #### MNIST->USPS
        
        x_train, y_train, x_test, y_test = mnist.load_mnist()
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
    elif TASK==5:
        #raise Exception('Not implemented/tested')
        #### MNIST -> SVHN
        
        x_train, y_train, x_test, y_test = mnist.load_mnist()
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
    elif TASK==6:
        raise Exception('Not implemented/tested')
        x_source=x_chexpert
        y_source=y_chexpert
        x_target=x_chest14
        y_target=y_chest14
    else: 
        raise Exception('Task '+str(task)' does not exist')
    return x_source, y_source, x_target, y_target