import random
####### label shifting and plotting
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', size=12, family='serif')
#class_label=1
#summand=1
#delta=0.7

## Note: this assumes that you have a one hot encoding and that class_label is within the range of that vector length
def label_shift(X,y,delta,class_label):
    ## takes a dataset and shifts the class label distribution by some margin delta 
    ## we will just give the ``removed'' entries back as the target distribution
    N=len(y)
    idx=[]
    y_2=[]
    X_2=[]
    x_target=[]
    y_target=[]
    #### initialise the random number generator? Necessary?
    np.random.seed()
    for i in range(N):
        if(y[i][class_label]==1):
            idx.append(i)
    ## what proportion of the label to remove
    M=int(len(idx)*delta)
    ## choose randomly delta amount of sample to include in target
    chosen=random.sample(idx,M)
    for i in range(N):
        if(i in chosen):
            x_target.append(X[i])
            y_target.append(y[i])
        else:
            y_2.append(y[i])
            X_2.append(X[i])
    x_target=np.array(x_target)
    y_target=np.array(y_target)
    X_2=np.array(X_2)
    y_2=np.array(y_2)
    return([X_2, y_2, x_target, y_target])

## Note: this assumes that you have a one hot encoding and that class_label is within the range of that vector length
def label_shift_linear(X,y,delta,labels,decreasing=True):
    """
    X: data points
    y: labels
    delta: percentage amount which you want to decrease for each label, i.e. slope for the shifting; delta in [0,1)
    labels: a vector of the possible labels i.e. for MNIST we have labels=[0,1,2,3,4,5,6,7,8,9]
    decreasing: bool to see if you want to make the shift increasing or decreasing
    """
    ## takes a dataset and shifts the class label distribution for all labels by
    ## a linearly increasing or decreasing amount
    ## we will just remove entries of the class for now
    
    L=len(labels)
    y_2=y
    X_2=X
    x_target=[]
    y_target=[]
    ## for every label go through and remove delta*label(or delta*(L-label)) amount of them (+1 to ensure overlap)
    for label in labels:
        if decreasing:
            delta2=delta*(label+1)
        else:
            delta2=delta*(L-label+1)
        assert(delta2<1)
        X_2, y_2, x_target2, y_target2=label_shift(X_2,y_2,delta2,label)
        if (label==0):
            x_target=x_target2
            y_target=y_target2
        else:   
            x_target=np.concatenate((x_target,x_target2))
            y_target=np.concatenate((y_target,y_target2))
        #print("--------------")
        #T=np.array(x_target)
        #print(T.shape)
    return([X_2, y_2, x_target, y_target])

def plot_labeldist(labels,y_1,label_1,save=False):
    """
    labels: a vector of the possible labels i.e. for MNIST we have labels=[0,1,2,3,4,5,6,7,8,9]
    y_1: labels of dataset
    label_1: name of dataset
    """
    
    import matplotlib.pyplot as plt


    # calculate the amount of label j in both datasets
    N=len(y_1)
   # M=len(labels)
    densities_1=[]
    sum=0
    for j in labels:
        sum=0
        for i in range(len(y_1)):
            if y_1[i][j]==1:
                sum+=1
        densities_1.append(sum)
       
    densities_1_rel=[]
   
    ## calculate relative density
    for i in range(len(densities_1)):
        densities_1_rel.append(densities_1[i]/(N))
        
    #print(densities_1_rel)


    width = 0.35       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels, densities_1_rel, width, label=label_1)
    #ax.set_ylim([0,1.25])
    ax.set_xlabel('Labels')
    ax.set_title('Label distribution')
    ax.legend()
    
    plt.show()
    if save:
        plt.savefig("labelshift.png",dpi=600)
    
def plot_splitbars(labels,y_1,y_2,label_1,label_2,save=False):
    """
    labels: a vector of the possible labels i.e. for MNIST we have labels=[0,1,2,3,4,5,6,7,8,9]
    y_1: labels of dataset1
    y_2: labels of dataset2
    label_1: name of dataset1
    label_2: name of dataset2
    """
    
    import matplotlib.pyplot as plt


    # calculate the amount of label j in both datasets
    densities_1=[]
    densities_2=[]
    sum=0
    for j in labels:
        sum=0
        for i in range(len(y_1)):
            if y_1[i][j]==1:
                sum+=1
        densities_1.append(sum)
        sum=0
        for i in range(len(y_2)):
            if y_2[i][j]==1:
                sum+=1
        densities_2.append(sum)
    densities_1_rel=[]
    densities_2_rel=[]
    ## calculate relative densities #### TODO: is this normalisation really what we want?
    for i in range(len(densities_1)):
        densities_1_rel.append(densities_1[i]/(densities_1[i]+densities_2[i]))
        densities_2_rel.append(densities_2[i]/(densities_1[i]+densities_2[i]))
    #print(densities_1_rel)
    #print(densities_2_rel)


    width = 0.35       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels, densities_1_rel, width, label=label_1)
    ax.bar(labels, densities_2_rel, width , bottom=densities_1_rel,
           label=label_2)
    ax.set_ylim([0,1.25])
    ax.set_xlabel('Labels')
    ax.set_title('Label distribution')
    ax.legend()
    #plt.show()
    if save:
        plt.savefig("labelshift.png",dpi=600)
    
def make_splits(y_source,x_shift,save=False):
    ##### make index lists for train and test splits for source

    #N=70000 ideally, however, how can we ensure this when the amount differs between labels?
    ## maybe we do not care too much about it exactly
    N=len(y_source)
    M=len(x_shift)
    train2=[]
    train1=[]
    test1=[]
    test2=[]
    ntr1=round(0.8*M)
    ntr2=round(0.8*N)
    ### sample n_tr datapoints (w/o replacement) to be the training set at random
    ## do this 10 times and save the indices into a file along with the training ones
    for i in range(10):
        T=random.sample(range(M),M)
        T2=random.sample(range(N),N)
        train1.append(T[:ntr1])
        train2.append(T2[:ntr2])
        test1.append(T[ntr1:])
        test2.append(T2[ntr2:])

    task1=[train1,test1]
    task2=[train2,test2]
    if save:
        import pickle
        pkl_file=open('splits_task1.pkl','wb')
        listoflist=task1
        pickle.dump(listoflist,pkl_file)

        pkl_file=open('splits_task2.pkl','wb')
        listoflist=task2
        pickle.dump(listoflist,pkl_file)