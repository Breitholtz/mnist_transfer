def make_splits(y_source,x_shift,save=False):
    import random
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

