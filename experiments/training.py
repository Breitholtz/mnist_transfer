import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

import os

    
def init_FC_model(binary=True):
    ### same as Dziugaite, to compare with rivasplata et al. in their case
    
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
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer,Dense
    model = Sequential()
    model.add(Dense(flattened_size))#InputLayer(input_shape=(flattened_size)))
    if binary:
        model.add(Dense(2,activation="softmax"))
    else:
        model.add(Dense(10, activation='softmax'))
    return model
    
## implement LeNet-5 architecture
def init_model():
    model = Sequential()
    model.add(Conv2D(6,(5,5),strides=(1,1), activation='tanh',input_shape=(32,32,1))) ## 6 5x5 conv kernels
    model.add(AveragePooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(16,(5,5),strides=(1,1), activation='tanh')) ## 16 5x5 conv kernels
    model.add(AveragePooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh'))
    model.add(Flatten())
    #model.add(Dense(120, activation='tanh'))  #equivalent to the last conv2d above?
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax')) # output layer
    return model
## implement LeNet-5-like architecture
def init_svhn_model(binary):
   
# We use stochastic gradient descent with
# momentum: 0.9
# weight_decay: 0.0005
# ------ this was not used on mnist<->svhn tests
# and the learning rate annealing described by the following formula:
# µp =µ0/(1 + α · p)^β,
# where p is the training progress linearly changing from 0
# to 1, µ0 = 0.01, α = 10 and β = 0.75 (the schedule
# was optimized to promote convergence and low error on
# the source domain).
# ------

# Following (Srivastava et al., 2014) we also use dropout and
# l_2-norm restriction when we train the SVHN architecture.
   
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

## implement LeNet-5-like architecture for mnist
def init_mnist_model(binary):
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
    ### we here take in the task number and architecture
    ### We return the model which fits the task
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
## shamelessly taken from : https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras

## Custom callback to be able to evaluate and save the results from several validation sets during training
class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)

            for i, result in enumerate(results):
                
                if i == 0:
                    valuename = validation_set_name + '_loss'
                else:
                    valuename = validation_set_name + '_' + self.model.metrics[i].name
                self.history.setdefault(valuename, []).append(result)

def train_model(model="SVHN" ,batch_size=128 ,total_epochs=25 ,iterations=1 ,x_train=[] ,y_train=[] ,x_test=[] ,y_test=[] ,x_target=[] ,y_target=[]):
    
    
    history = AdditionalValidationSets([(x_target, y_target, 'target_val')])
    
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "checkpoints/"+model+"-cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
        ## tune when to save as needed for plots
    save_freq=10*235    ### 469 = ceiling(60000/128) i.e training set for MNIST/MNIST-M
    )
    
    histories=[]
    M=init_SVHN_model()
    for i in range(iterations):
        if model=="SVHN":
            M=init_SVHN_model()
        elif model=="MNIST":
            M=init_MNIST_model()
        elif model=="MNIST-M":
            M=init_MNIST_model()
        elif model=="2MNIST-M":
            M=init_MNIST_model()
        elif model=="binary":
            M=init_MNIST_model_binary()
        # Save the weights using the `checkpoint_path` format
        M.save_weights(checkpoint_path.format(epoch=0))
        ## choose loss function, optimiser etc. and train
        M.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                      metrics=['accuracy'],)

        fit_info = M.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=total_epochs,
           verbose=1,
           validation_data=(x_test, y_test),
           callbacks=[history,cp_callback])
        histories.append(history.history)
    return histories
