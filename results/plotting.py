import matplotlib.pyplot as plt
def plot_results(total_epochs,model="SVHN",result=[], xlabel="Epoch",save=True):#,ylabel="", title=""): 
    ### TODO: do one for each type of plot?
    
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY_H:M
    dt_string = now.strftime("%d%m%Y_%H:%M")
    #print("date and time =", dt_string)
    
    ## plotting and saving to disk
    x=[i+1 for i in range(total_epochs)]

    if model=="SVHN":
        prefix=["S2MTGTACC","S2MVAL","S2MLOSS","S2MVALLOSS","S2MTGTLOSS"]
    elif model=="MNIST":
        prefix=["M2STGTACC","M2SVAL","M2SLOSS","M2SVALLOSS","M2STGTLOSS"]
    elif model=="MNIST-M":
         prefix=["MM2MTGTACC","MM2MVAL","MM2MLOSS","MM2MVALLOSS","MM2MTGTLOSS"]
    elif model=="2MNIST-M":
        prefix=["M2MMTGTACC","M2MMVAL","M2MMLOSS","M2MMVALLOSS","M2MMTGTLOSS"]

    f, ax = plt.subplots()
    ax.plot(x,result['target_val_accuracy'], '*-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Target acc'], loc = 0)
    ax.set_title('Target acc per epoch')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    if(save):
        f.savefig("./images/"+prefix[0]+dt_string+".pdf")
   

    f, ax = plt.subplots()
    ax.plot(x,result['val_accuracy'], 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Validation acc'], loc = 0)
    ax.set_title('Validation acc per epoch')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    if(save):
        f.savefig("./images/"+prefix[1]+dt_string+".pdf")

    f, ax = plt.subplots()
    
    ax.plot(x,result['loss'], 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Training loss'], loc = 0)
    ax.set_title('Training loss per epoch')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Loss')
    if(save):
        f.savefig("./images/"+prefix[2]+dt_string+".pdf")

    f, ax = plt.subplots()
    ax.plot(x,result['val_loss'], 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Validation loss'], loc = 0)
    ax.set_title('Validation loss per epoch')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Loss')
    if(save):
        f.savefig("./images/"+prefix[3]+dt_string+".pdf")
    
    f, ax = plt.subplots()
    ax.plot(x,result['target_val_loss'], 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Target acc'], loc = 0)
    ax.set_title('Target loss per epoch')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Loss')
    if(save):
        f.savefig("./images/"+prefix[4]+dt_string+".pdf")
        
        
        
def plot_results_data(sizes=[],model="SVHN",result=[],save=True):# plot for the increasing amount of data
    ### TODO: do one for each type of plot?
    
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY_H:M
    dt_string = now.strftime("%d%m%Y_%H:%M")
    #print("date and time =", dt_string)
    
    ## plotting and saving to disk
    x=[i+1 for i in range(total_epochs)]

    if model=="SVHN":
        prefix=["S2MTGTACC","S2MVAL","S2MVALLOSS","S2MTGTLOSS"]
        target="MNIST"
    elif model=="MNIST":
        prefix=["M2STGTACC","M2SVAL","M2SVALLOSS","M2STGTLOSS"]
        target="SVHN"
    elif model=="MNIST-M":
         prefix=["MM2MTGTACC","MM2MSRCACC","MM2MSRCLOSS","MM2MTGTLOSS"]
    elif model=="2MNIST-M":
        prefix=["M2MMTGTACC","M2MMVAL","M2MMLOSS","M2MMVALLOSS","M2MMTGTLOSS"]
    
    f, ax = plt.subplots()
    ax.plot(sizes,aggregate_acc_t, '*-')
    # Plot legend and use the best location automatically: loc = 0.
    #ax.legend(['Train acc', 'Validation acc','Target acc'], loc = 0)
    ax.legend(['Target acc'], loc = 0)
    #ax.legend(['Validation acc'], loc = 0)
    ax.set_title('Target acc per amount of training data ('+model+'->'+target+')')
    #ax.set_title('Validation acc per epoch')
    ax.set_xlabel('Data amount')
    ax.set_ylabel('Accuracy')
    if(save):
        f.savefig("./images/data/"+prefix[0]+dt_string+".pdf")

    f, ax = plt.subplots()
    #ax.plot(x,result['accuracy'], 'o-')
    #ax.plot(x,result['val_accuracy'], 'x-')
    ax.plot(sizes,aggregate_acc_s, '*-')
    # Plot legend and use the best location automatically: loc = 0.
    #ax.legend(['Train acc', 'Validation acc','Target acc'], loc = 0)
    ax.legend(['Source acc'], loc = 0)
    #ax.legend(['Validation acc'], loc = 0)
    ax.set_title('Source acc per amount of training data ('+model+'->'+target+')')
    #ax.set_title('Validation acc per epoch')
    ax.set_xlabel('Data amount')
    ax.set_ylabel('Accuracy')
    if(save):
        f.savefig("./images/data/"+prefix[1]+dt_string+".pdf")

    f, ax = plt.subplots()
    #ax.plot(x,result['accuracy'], 'o-')
    #ax.plot(x,result['val_accuracy'], 'x-')
    ax.plot(sizes,aggregate_loss_s, '*-')
    # Plot legend and use the best location automatically: loc = 0.
    #ax.legend(['Train acc', 'Validation acc','Target acc'], loc = 0)
    ax.legend(['Source loss'], loc = 0)
    #ax.legend(['Validation acc'], loc = 0)
    ax.set_title('Source loss per amount of training data ('+model+'->'+target+')')
    #ax.set_title('Validation acc per epoch')
    ax.set_xlabel('Data amount')
    ax.set_ylabel('Loss')
    if(save):
        f.savefig("./images/data/"+prefix[2]+dt_string+".pdf")
    
    f, ax = plt.subplots()
    #ax.plot(x,result['accuracy'], 'o-')
    #ax.plot(x,result['val_accuracy'], 'x-')
    ax.plot(sizes,aggregate_loss_t, '*-')
    # Plot legend and use the best location automatically: loc = 0.
    #ax.legend(['Train acc', 'Validation acc','Target acc'], loc = 0)
    ax.legend(['Target loss'], loc = 0)
    #ax.legend(['Validation acc'], loc = 0)
    ax.set_title('Target loss per amount of training data ('+model+'->'+target+')')
    #ax.set_title('Validation acc per epoch')
    ax.set_xlabel('Data amount')
    ax.set_ylabel('Loss')
    if(save):
        f.savefig("./images/data/"+prefix[3]+dt_string+".pdf")