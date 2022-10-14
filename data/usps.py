import numpy as np
import gzip
import pickle
import tensorflow.keras as keras
def load_usps():
    """
    function which loads 28x28 USPS images and makes them into a usable format
    """
    num_classes = 10
    
    ## copied and changed from https://github.com/JingWang18/Discriminative-Feature-Alignment/
    f = gzip.open('usps_28x28.pkl', 'rb')
    data_set = pickle.load(f, encoding="bytes")
    f.close()
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    
    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 28, 28, 1))
    img_test = img_test.reshape((img_test.shape[0], 28, 28, 1))

    img_train = np.pad(img_train,((0,0),(2,2),(2,2),(0,0))) #padding to make images 32x32 and not 28x28
    img_test = np.pad(img_test,((0,0),(2,2),(2,2),(0,0))) 
    
    ## normalising to unit variance
    sigma=np.std(img_train)
    img_train /= sigma 
    img_test /= sigma

    ## mean subtraction
    mu=np.mean(img_train)
    img_train -= mu
    img_test -= mu
    print('mean, variance', mu, sigma)
    ## make labels into categorical classes
    label_train = keras.utils.to_categorical(label_train, num_classes)
    label_test = keras.utils.to_categorical(label_test, num_classes)
    
    ## expand to (N,32,32,3) so that we can compare the two datasets
 
#     img_train=np.concatenate((img_train,img_train,img_train),axis=3)
#     img_test=np.concatenate((img_test,img_test,img_test),axis=3) 
    
    
    
    print("---------------Load USPS----------------")
    print('Training set', img_train.shape, label_train.shape)

    print('Test set', img_test.shape, label_test.shape)
    print("\n")
    return img_train, label_train, img_test, label_test
