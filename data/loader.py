import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.image import resize
from PIL import Image
from data import tasks
#from skimage.transform import resize

#source_image_dir2="/home/users/adam/Code/Datasets/"
source_image_dir="/cephyr/NOBACKUP/groups/snic2021-23-538/"
class CheXpertDataGenerator(Sequence):
    'Data Generator for CheXpert and chestXray14'
    
    def __init__(self, dataset_df, y, batch_size=16,
                 target_size=(224, 224),  verbose=0,
                 shuffle_on_epoch_end=False, random_state=1):
        self.dataset_df = dataset_df
        self.y=y
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.x_path=dataset_df["Path"]
        self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
    def __bool__(self):
        return True

    def __len__(self):
        return self.steps
    def __getitem__(self, idx):
        # print('idx....', idx)
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path]).astype(np.float32)
        batch_x = self.transform_batch_images(batch_x)
        y=tasks.binarize(self.y,0) ### set no finding as the 1
        batch_y = y[idx * self.batch_size:(idx + 1) * self.batch_size]
        #print(batch_y)
        return (batch_x, batch_y)

    def load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir, image_file)
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = tf.image.resize(image_array, self.target_size)
        return image_array

    def transform_batch_images(self, batch_x):
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        if batch_x.shape == imagenet_mean.shape:
            batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps*self.batch_size, :]

    #def prepare_dataset(self):
        ### here we make the dataset as desired
#         self.dataset_df = self.dataset_df[self.dataset_df['Frontal/Lateral'] == 'Frontal']
#         df = self.dataset_df.sample(frac=1., random_state=self.random_state)
#         df.fillna(0, inplace=True)
#         self.x_path, y_df = df["Path"].as_matrix(), df[self.class_names]  

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
