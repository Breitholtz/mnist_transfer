import tensorflow as tf
import numpy as np
import collections

def tf_load_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image=tf.cast(image, tf.float32)
    return image
def tf_read_and_resize_image(filename,img_size):
    ### takes file path and image size and resizes the image
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image=tf.cast(image, tf.float32)
    image = tf.image.resize(image, (img_size, img_size))#, method=tf.image.ResizeMethod.BILINEAR)
    return image
def binarize(y,x):
    ## take in one hot label encoding and make it into either 'label x' or 'not label x'
    ## x is in [0,5] as we have at least 6 overlapping labels in our chestxray data
    ## labeldict={"No Finding":0,"Cardiomegaly":1,"Edema":2,"Consolidation":3,"Atelectasis":4,"Effusion":5}
    y_new=[]
    mask=np.zeros(6)
    mask[x]=1
    for i in y:
        if np.dot(i,mask)==1:
            ## we have the label
            y_new.append([0,1])
        else:
            ## we do not have the label
            y_new.append([1,0])
    return np.array(y_new)
def load_resize_and_save(chexpert=False,img_size=32):
##### loads the xray datasets from the raw images and then resizes to desired size and saves them in a new directory
    if chexpert:
        data_path="/home/adam/Code/Datasets/chexpert/"
        _DATA_DIR = "/home/adam/Code/Datasets/chexpert/CheXpert-v1.0-small" ### where is the source data?
        _TRAIN_DIR = os.path.join(_DATA_DIR, "train")
        _VALIDATION_DIR = os.path.join(_DATA_DIR, "valid")
        _TRAIN_LABELS_FNAME = os.path.join(_DATA_DIR, "train.csv")
        _VALIDATION_LABELS_FNAME = os.path.join(_DATA_DIR, "valid.csv")
        
        filenames,labels=load_from_csv(data_path,_TRAIN_LABELS_FNAME)
        
        filenames_2,labels_2=load_from_csv(data_path,_VALIDATION_LABELS_FNAME)

        filenames.extend(filenames_2)
        labels.extend(labels_2)
     
        new_path=data_path+"resized"+str(img_size)+"chex"
#         if not os.path.exists(new_path):
#             os.makedirs(new_path)
        ## load in, resize and save image in array file
        for file in filenames:
            img=np.array(tf_read_and_resize_image(file,img_size))
            
            A=file.split('/')[-4:]
            path=""
            for a in A:
                if a[-4:]==".jpg":
                    if not os.path.exists(new_path+path):
                        os.makedirs(new_path+path)
                    
                path+="/"+a
                
            output_path=new_path+path
            
            #print(output_path)
            tf.keras.preprocessing.image.save_img(output_path,img)
            
        #np.save("/home/adam/Code/Datasets/chexpert/chexpert128.npy",np.array([np.array(tf_read_and_resize_image(file)) for file in filenames]))
        #np.save("/home/adam/Code/Datasets/chexpert/labels.npy",np.array(labels)) # save labels to separate file
    else:
        data_path="/home/adam/Code/Datasets/chestXray14/"
            os.makedirs(new_path)

        dirs = [l for l in os.listdir(data_path+"images/") if l != '.DS_Store']
        for file in dirs:
            img=np.array(tf_read_and_resize_image(data_path+"images/"+file,img_size))
            #print(img)
            tf.keras.preprocessing.image.save_img(new_path+"/"+file,img)



def load_to_array(chexpert=False,img_size=32):
    if chexpert:
        data_path="/home/adam/Code/Datasets/chexpert/"
        _DATA_DIR = "/home/adam/Code/Datasets/chexpert/CheXpert-v1.0-small" 
        _TRAIN_DIR = os.path.join(_DATA_DIR, "train")
        _VALIDATION_DIR = os.path.join(_DATA_DIR, "valid")
        _TRAIN_LABELS_FNAME = os.path.join(_DATA_DIR, "train.csv")
        _VALIDATION_LABELS_FNAME = os.path.join(_DATA_DIR, "valid.csv")

        filenames,labels=load_from_csv(data_path,_TRAIN_LABELS_FNAME,True)

        filenames_2,labels_2=load_from_csv(data_path,_VALIDATION_LABELS_FNAME,True)
        
        filenames.extend(filenames_2)
        arr=[]
        labels.extend(labels_2)
        #for file in filenames:
            #print(np.array(tf_load_image(file)))
            #sys.exit(-1)
        arr= np.array([np.array(tf_load_image(img)) for img in filenames])
        np.save(data_path+"chexpert_"+str(img_size)+".npy",arr)
        np.save(data_path+"chexpert_"+str(img_size)+"_labels.npy",labels)
        return arr, np.array(labels)
    else:
        data_path="/home/adam/Code/Datasets/chestXray14/"
        dirs = [l for l in os.listdir(data_path+"resized"+str(img_size)+"/") if l != '.DS_Store']
        arr=[]
        #arr= np.array([np.array(tf_load_image(data_path+"resized"+str(img_size)+"/"+img)) for img in dirs])
        y=make_xray14_labels()
        #np.save(data_path+"chestxray14_"+str(img_size)+".npy",arr)
        np.save(data_path+"chestxray14_"+str(img_size)+"_labels.npy",y)
        return arr, np.array(y)    
    
    
    
def load_from_csv(imgs_path,csv_path,resized=False):
    _LABELS = collections.OrderedDict({
            "-1.0": "uncertain",
            "1.0": "positive",
            "0.0": "negative",
            "": "unmentioned",
        })
    labeldict={"positive":1,"negative":0, "unmentioned":0,"uncertain":1} ### sets all uncertain labels to 1
    overlapping_labels=[0,2,5,6,8,10] ## according to pham et al. NF,CM,ED,CD,AC,PE
    ##### loads chexpert filenames and labels from files
    label_arr=[]
    arr=[]
    with tf.io.gfile.GFile(csv_path) as csv_f:
        reader = csv.DictReader(csv_f)
        # Get keys for each label from csv
        label_keys = reader.fieldnames[5:]

        for row in reader:
            # Get image based on indicated path in csv
            name = row["Path"]
            labels = [_LABELS[row[key]] for key in label_keys]
            labels_overlap=[labeldict[labels[i]] for i in overlapping_labels]
            if resized:
                A=name.split('/')[1:]
                path="resized32chex"
                for a in A:
                    path+="/"+a
                name=path
                
                    
            ## save the image_name and the label array
            label_arr.append(labels_overlap)
            #print(name)
            
            arr.append(os.path.join(imgs_path, name))
        return arr,label_arr
    
def make_xray14_labels():
    ##### load the labels and convert the labels to binary vectors which maps the occurrence of a label to a 1
    data_path="/home/adam/Code/Datasets/chestXray14/"
    labeldict={"No Finding":0,"Cardiomegaly":1,"Edema":2,"Consolidation":3,"Atelectasis":4,"Effusion":5}
    data = pd.read_csv(data_path+"Data_Entry_2017_v2020.csv")
    sample = os.listdir(data_path+"resized32/")

    sample = pd.DataFrame({'Image Index': sample})

    sample = pd.merge(sample, data, how='left', on='Image Index')

    sample.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                      'Patient_Age', 'Patient_Gender', 'View_Position',
                      'Original_Image_Width', 'Original_Image_Height',
                      'Original_Image_Pixel_Spacing_X',
                      'Original_Image_Pixel_Spacing_Y']#, 'Unnamed']
    def make_one_hot(label_string):
        result=np.zeros(6)
        labels=label_string.split('|')
        for l in labels:
            if l not in ["No Finding","Cardiomegaly","Edema","Consolidation","Atelectasis","Effusion"]:
                pass
            else:
                result[labeldict[l]]=1
        return result.astype(int)

    sample['Finding_Labels'] = sample['Finding_Labels'].apply(lambda x: make_one_hot(x))
    #print(sample['Finding_Labels'].shape)
    y=sample['Finding_Labels']
    return np.array(y)
