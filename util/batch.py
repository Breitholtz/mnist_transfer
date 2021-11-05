import os, re

def get_job_args(task, bound='germain', alpha=0.1, sigma=[3,2], epsilon=[0.01], binary=False, n_classifiers=4):
 
    if binary:
        #with open('posteriors/'+"task"+str(task)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+'/params.txt', 'rb+') as f:
         #   params=f.readlines()
        #f.close()
        prior_path="priors/"+"task"+str(task)+"/Binary/"+str(int(100*alpha))+"/prior.ckpt"
        result_path="results/"+"task"+str(task)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"
    else:
        #with open('posteriors/'+"task"+str(task)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+'/params.txt', 'rb+') as f:
        #    params=f.readlines()
        #f.close()
        prior_path="priors/"+"task"+str(task)+"/"+str(int(100*alpha))+"/prior.ckpt"
        result_path="results/"+"task"+str(task)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"_"

    #epsilon=float(params[1]) # @TODO: Superfluous? Isn't this submitted?
    #epochs_trained=int(params[2]) # @TODO: Unused as far as I can see
     
    posterior_paths = posterior_checkpoints(task, epsilon, alpha, binary=binary)
    
    arg_list = []
    for post in posterior_paths: 
        args = {
            'task': task, 
            'prior_path': prior_path, 
            'posterior_path': post,
            'bound': bound, 
            'alpha': alpha,
            'sigma': sigma, 
            'epsilon': epsilon, 
            'binary': binary,
            'n_classifiers': n_classifiers
        }
        arg_list.append(args)
        
    return arg_list
    
def posterior_checkpoints(task, epsilon, alpha, binary=False):
    ### Here we do something more intelligent to not have to hardcode the epoch amounts. 
    ### we parse the filenames and sort them in numerical order and then load the weights
    if binary:
        base_path="posteriors/"+"task"+str(task)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
    else:
        base_path="posteriors/"+"task"+str(task)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))
        
    list1=[]
    list2=[]
    dirFiles = os.listdir(base_path) #list of directory files
    ## remove the ckpt.index and sort so that we get the epochs that are in the directory
    for files in dirFiles: #filter out all non jpgs
        if '.ckpt.index' in files:
            name = re.sub('\.ckpt.index$', '', files)
            ### if it has a one it goes in one list and if it starts with a two it goes in the other
            if (name[0]=="1"):
                list1.append(name)
            elif (name[0]=="2"):
                list2.append(name)
                
    list1.sort(key=lambda f: int(re.sub('\D', '', f)))
    num_batchweights=len(list1)
    list2.sort(key=lambda f: int(re.sub('\D', '', f)))
    list1.extend(list2)
    Ws=list1
        
    path="posteriors/"+"task"+str(task)+"/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"/"#"{epoch:0d}.ckpt"
    if binary:
        path="posteriors/"+"task"+str(task)+"/Binary/"+str(int(1000*epsilon))+"_"+str(int(100*alpha))+"/"#"{epoch:0d}.ckpt"
    
    posterior_paths = [os.path.join(path, str(checkpoint)+".ckpt") for checkpoint in Ws]
    
    return posterior_paths