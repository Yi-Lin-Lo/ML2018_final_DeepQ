import pandas as pd
import numpy as np
from myModelFactory_8 import ModelFactory
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import CSVLogger, ModelCheckpoint
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import cv2
import os
from sklearn.metrics.ranking import roc_auc_score
from callback_5 import MultipleClassAUROC, MultiGPUModelCheckpoint
from sklearn.metrics import roc_auc_score
import time
import random
from imgaug import augmenters as iaa
import sys
img_path = sys.argv[1] + '/'
train_path = sys.argv[2]

np.random.seed(666)

def add_salt_pepper_noise(X_img):
    # Need to produce a copy as to not modify the original image
    X_img_copy = X_img.copy()
    row, col, _ = X_img_copy.shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_img_copy.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_img_copy.size * (1.0 - salt_vs_pepper))
    
 
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img_copy.shape]
    X_img_copy[coords[0], coords[1], :] = 1

        # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img_copy.shape]
    X_img_copy[coords[0], coords[1], :] = 0
    return X_img_copy

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]



def augmentation(X,Y):

    datagen = ImageDataGenerator(
        rotation_range = 15,
        #width_shift_range = 20, #[-10,10]
        #height_shift_range = 20,
        shear_range = 0.01,
        zoom_range = 0.3, #0.9 ~ 1.1
        horizontal_flip = True,
        #fill_mode = 'nearest'
        #featurewise_center=True,
        #featurewise_std_normalization=True 
        
    )
    datagen.fit(X)
    a,b = next( datagen.flow(X, Y, batch_size= X.shape[0] ) )
    gen_data = []
    seq = iaa.Sequential([iaa.CoarseDropout(p=0.05,size_percent=0.3)])
    #seq_2 = iaa.Sequential([iaa.GaussianBlur(sigma=(0, 0.8))])
    seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
    #seq_det_2 = seq_2.to_deterministic()
    #a = seq_det.augment_images(a)
    crop_size = (299,299)
    for i in range(a.shape[0]):
        randnum = random.random()
        img_tmp = random_crop(a[i],crop_size)
        if randnum > 0.5:
            gen_data.append( img_tmp )
        elif randnum > 0.3:
            gen_data.append(add_salt_pepper_noise(img_tmp))
        else:
            gen_data.append(seq_det.augment_image((img_tmp)*255 )/255.0)
        #else:
        #    gen_data.append(seq_det_2.augment_image((img_tmp)*255 )/255.0)

    gen_data=np.array(gen_data)
    #print(gen_data.shape)
    return gen_data,b


def getValidData(file, valid_range):
    with open(file) as f:
        line_num = 0
        X, Y = [],[]
        for line in f:
            if line_num >= valid_range[0]+1 and line_num < valid_range[1]+1:
                x,y = process_line(line)
                crop_size = (299,299)
                x = random_crop(x,crop_size)  #
                X.append(x)
                Y.append(y)
            line_num = line_num +1
        return ( np.array(X),np.array(Y) )

def process_line(line):
    path = line[:line.find(',')]
    label = line[line.find(',')+ 1:line.find(',') +28]
    label = np.array( list( map( int ,label.split(' ') ) ) )
    load_size = 331
    img = cv2.imread(img_path+ path)
    img = ( cv2.resize(img, (load_size, load_size))  ).astype(np.float32)  /255.0
    return img, label

def load_data_from_file(file,batch_size, valid_range):
    n_data = 1000
    cnt = 0
    X,Y = [],[]
    while True:
        with open(file) as f:
            line_num = 0
  
            for line in f:
                if (line_num <(valid_range[0]+1) or line_num > (valid_range[1]+1)) and line_num < 10003 and line_num >0:
                    x,y = process_line(line)      
                    X.append(x)
                    Y.append(y)
                    cnt += 1
                    if cnt == batch_size:
                        X = np.array(X)
                        Y = np.array(Y)
                        (auged_X, auged_Y) = augmentation(X,Y)

                        yield ( auged_X, auged_Y )
                        cnt = 0
                        X, Y = [],[]
                line_num+=1 
                
valid_range = [3200, 4000]

valid_x, valid_y = getValidData(train_path, valid_range )



model_factory = ModelFactory()
base_model_name="InceptionResNetV2"
use_base_model_weights=True
class_names=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']
model = model_factory.get_model(
            class_names,
            model_name=base_model_name,
            use_base_weights=use_base_model_weights,
            )

layer_cnt = 0
for layer in model.layers:
    #print(layer_cnt)
    #print(layer.name)
   # if layer_cnt <=6:
   #     layer.trainable = False
    layer_cnt = layer_cnt +1

    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer= regularizers.l2(0.05)
        #layer.kernel_regularizer= regularizers.l1(0.001)
        #activity_regularizer=regularizers.l1(0.001)
        layer.Dropout = 0.2

#model.summary()
model.save('./model/weight.h5')


model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['acc'])

auroc = MultipleClassAUROC(
            base_model=model,
            sequence=valid_x,
            valid_y = valid_y,
            class_names=class_names,
            weights_path='./model/weight.h5',

        )


batch_size = 12
num_epochs = 30

callbacks = [auroc]

modelcheckpoint = ModelCheckpoint('./model/unuse.h5',mode='min',monitor='val_loss',save_best_only=True)

callbacks.append(modelcheckpoint)
csv_logger = CSVLogger('vgg_log.csv', separator=',', append=False)
callbacks.append(csv_logger)

class_weight = {0: 1.,
                1: 1.,
                2: 1.,
                3: 2.,
                4: 1.,
                5: 2.,
                6: 1.,
                7: 1.,
                8: 1.,
                9: 1.,
                10: 2.,
                11: 1.,
                12: 2.,
                13: 10.,
                }


model.fit_generator( generator = load_data_from_file(train_path, batch_size, valid_range ) 
                    ,  steps_per_epoch = ((10002 -800 + batch_size - 1) // batch_size)
                    , epochs = num_epochs
                    , validation_data=(valid_x, valid_y)
                   ,callbacks=callbacks
                   ,class_weight=class_weight)
