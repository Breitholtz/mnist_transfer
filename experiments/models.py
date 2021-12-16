import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

import os

    
def init_FC_model(binary=True):
    """
    Fully connected model, similar to rivasplata et al., dziugaite et al. etc.
    """
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential()
    model.add(Dense(1024,input_shape=(32,32,3), activation = 'relu'))
    #model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
    model.add(Dense(600, activation = 'relu'))
    #model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
    model.add(Dense(600, activation = 'relu'))
    model.add(Flatten())
    if binary:
        model.add(Dense(2, activation = 'softmax'))
    else:
        model.add(Dense(10, activation = 'softmax'))
    return model
def init_lr_model(flattened_size,binary=True):
    """
    Logistic regression model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer,Dense
    model = Sequential()
    model.add(Dense(flattened_size))#InputLayer(input_shape=(flattened_size)))
    if binary:
        model.add(Dense(2,activation="softmax"))
    else:
        model.add(Dense(10, activation='softmax'))
    return model
    
## implement LeNet-5-like architecture
def init_svhn_model(binary):
    """  
    Model used in Dziugaite for training on SVHN
     SGD with:
         momentum: 0.9
         weight_decay: 0.0005
         dropout
         l2 regularization
    """
    model = Sequential()
    model.add(Conv2D(64,(5,5),strides=(1,1), activation='relu',input_shape=(32,32,3),kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005))) ## 6 5x5 conv kernels
    model.add(Dropout(0.9))
    model.add(AveragePooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(64,(5,5),strides=(1,1), activation='relu',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005))) ## 16 5x5 conv kernels
    model.add(Dropout(0.75))
    model.add(AveragePooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(128,(5,5),strides=(1,1), activation='relu',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005)))
    model.add(Dropout(0.75))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(3072, activation='relu',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005)))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005)))
    model.add(Dropout(0.5))
    if binary:
        model.add(Dense(10, activation='softmax',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005))) # output layer
    else:
        model.add(Dense(10, activation='softmax',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005))) # output layer
    
    return model


def init_mnist_model(binary):
    """
    LeNet-5 type model 
    """
    model = Sequential()
    model.add(Conv2D(32,(5,5),strides=(1,1), activation='relu',input_shape=(32,32,3))) ## 6 5x5 conv kernels
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
    model.add(Conv2D(48,(5,5),strides=(1,1), activation='relu')) ## 16 5x5 conv kernels
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    if binary:
        model.add(Dense(2, activation='softmax')) # output layer
    else:
        model.add(Dense(10, activation='softmax')) # output layer
    return model

def init_task_model(TASK=2,binary=True,arch="lenet"): 
    """
     Function that takes in the task number and architecture
     
     It returns the model which fits the task
    """
    if arch not in ["lr","lenet","fc","resnet"]:
        raise Exception('Architecture '+arch+' not implemented/tested')
    if TASK==1 or TASK==2 or TASK==3 or TASK==4:
        #### MNIST label shift (1) and mix of MNIST and MNIST-M (2)
        if arch=="lr":
            model=init_lr_model(binary)
        elif arch=="lenet":
            model=init_mnist_model(binary)
        elif arch=="fc":
            model=init_fc_model(binary)
        else:
            model=init_resnet_model(binary)
    elif TASK==5:
        
        #### MNIST -> SVHN
        if arch=="lr":
            model=init_lr_model(binary)
        elif arch=="lenet":
            model=init_mnist_model(binary)
        elif arch=="fc":
            model=init_fc_model(binary)
        else:
            model=init_resnet_model(binary)
    elif TASK==6:
        #### CheXpert -> chestxray14
        if arch=="lr":
            model=init_lr_model(binary)
        elif arch=="lenet":
            model=init_mnist_model(binary)
        elif arch=="fc":
            model=init_fc_model(binary)
        else:
            model=init_resnet_model(binary)
    else:
        raise Exception('Task '+str(TASK)+' not implemented/tested')
    return model
