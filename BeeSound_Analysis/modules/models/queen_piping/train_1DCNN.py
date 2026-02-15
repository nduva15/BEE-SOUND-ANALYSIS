from __future__ import print_function
import keras
#from keras.datasets import mnist
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, BatchNormalization, Activation
#from keras.optimizers import RMSprop 
#from keras.layers import Embedding

import numpy as np
import scipy.io as sio
import my_kerasloader as kl
import tensorflow as tf
from sklearn.metrics import accuracy_score

## reduce tensorflow verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import classification as clf
#import functions as fc
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#import matplotlib.pyplot as plt
#import my_tools as mt
#from IPython.display import display, Audio

from tensorflow import keras


## load data
filemat = '___tmp.mat';

############## load data
fp = sio.loadmat(filemat);


#some global variables
SAMPLING_RATE = 22050
#VALID_SPLIT = 0.1
#TEST_SPLIT = 0.3
#SHUFFLE_SEED =  43
#SAMPLES_TO_DISPLAY = 10
#SCALE = 0.5

#BATCH_SIZE = 128

BATCH_SIZE = 16
EPOCHS = 25


x_train = fp['x_train'];
y_train = fp['y_train']-1;
model_name = fp['model_name'];

N = x_train.shape[1];
num_classes = np.unique(y_train).shape[0];


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes);
y_train = y_train[0];

## fix bug
y_train = y_train[:,0];

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))

################################### Build model here
model = clf.make_model_1DCNN((SAMPLING_RATE // 2, 1), 2)

# for debug only
#model.summary()

model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

#model_save_filename = model_name[0]+'.h5';
#earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True) #
#mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
#    model_save_filename, monitor="val_accuracy", save_best_only=True
#)

## training 
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS, verbose=0); #, callbacks = [earlystopping_cb]

#callbacks = [earlystopping_cb, mdlcheckpoint_cb]
                    
kl.save_keras_model(model, model_name[0]+'.json', model_name[0]+'.h5');

#verbose=1

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

#plt.plot(history.history['acc'])
#plt.title('training accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.savefig('acc.png')
#plt.close();

#plt.plot(history.history['loss'])
#plt.title('training loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.show()
#plt.savefig('loss.png')
#plt.close();






