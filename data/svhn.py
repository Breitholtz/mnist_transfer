def load_svhn():
    import h5py
    import numpy as np
    ### import svhn_cropped without grayscale

    # Open the file as readonly
    h5f = h5py.File('SVHN_cropped.h5', 'r')

    # Load the training, test and validation set
    x_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    x_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]
    x_extra = h5f['X_extra'][:]
    y_extra = h5f['y_extra'][:]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_extra = x_extra.astype('float32')

    #### make validation set from train and extra (and make extra part of train)

    x_train=np.append(x_train,x_extra, axis=0)
    y_train=np.append(y_train,y_extra, axis=0)
    ## normalising
    sigma=np.std(x_train)
    x_train /= sigma 
    x_test /= sigma


    mu=np.mean(x_train)
    x_train -= mu
    x_test -= mu

    print('mean, variance', mu, sigma)
    print("---------------Load SVHN----------------")
    print('Training set', x_train.shape, y_train.shape)
    #print('Extra set', X_extra.shape, Y_extra.shape)
    print('Test set', x_test.shape, y_test.shape)
    
    return  x_train, y_train, x_test, y_test
