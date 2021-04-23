def load_svhn():
    import h5py
    import numpy as np
    ### import svhn_cropped without grayscale

    # Open the file as readonly
    h5f = h5py.File('SVHN_cropped.h5', 'r')

    # Load the training, test and validation set
    X_train = h5f['X_train'][:]
    Y_train = h5f['y_train'][:]
    X_test = h5f['X_test'][:]
    Y_test = h5f['y_test'][:]
    X_extra = h5f['X_extra'][:]
    Y_extra = h5f['y_extra'][:]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_extra = X_extra.astype('float32')

    #### make validation set from train and extra (and make extra part of train(?))

    X_train=np.append(X_train,X_extra, axis=0)
    Y_train=np.append(Y_train,Y_extra, axis=0)
    ## normalising
    sigma=np.std(X_train)
    X_train /= sigma 
    X_test /= sigma


    mu=np.mean(X_train)
    X_train -= mu
    X_test -= mu





    print('mean, variance', mu, sigma)
    print("---------------Load SVHN----------------")
    print('Training set', X_train.shape, Y_train.shape)
    #print('Extra set', X_extra.shape, Y_extra.shape)
    print('Test set', X_test.shape, Y_test.shape)
    
    return  X_train, Y_train, X_test, Y_test
'''
## import svhn_cropped which is svhn in 32x32 size, grayscale
####### do not load this at the moment!
####### do not load this at the moment!
####### do not load this at the moment!
####### do not load this at the moment!
####### do not load this at the moment!
####### do not load this at the moment!
####### do not load this at the moment!
# Open the file as readonly
h5f = h5py.File('SVHN_gray.h5', 'r')

# Load the training, test and validation set
X_train = h5f['X_train'][:]
Y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
Y_test = h5f['y_test'][:]
X_val = h5f['X_val'][:]
Y_val = h5f['y_val'][:]

# Close this file
h5f.close()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_test = x_test.astype('float32')
## normalising
#_train[:,axis=3] /= 255.0 
#X_test /= 255.0
#X_val /= 255.0

print('Training set', X_train.shape, Y_train.shape)
print('Validation set', X_val.shape, Y_val.shape)
print('Test set', X_test.shape, Y_test.shape)
'''