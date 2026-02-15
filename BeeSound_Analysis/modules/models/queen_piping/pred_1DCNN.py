from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import RMSprop 
from keras.layers import Embedding

import numpy as np
import sys
import scipy.io as sio
import my_kerasloader as kl

from sklearn.metrics import accuracy_score
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

filemat = '___tmp.mat';
filemat2 = '___tmp2.mat';

model_name = sys.argv[1];


fp = sio.loadmat(filemat);

x_test  = fp['x_test'];
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

#N = x_test.shape[1];

model = kl.load_keras_model(model_name+'.json', model_name+'.h5');

#model.summary();

y_test_cat = model.predict(x_test); #batch_size=128

#y_test = y_test_cat;
y_test = np.argmax(y_test_cat,axis=1)+1

sio.savemat(filemat2, {"y_test": y_test,"y_test_cat": y_test_cat,});

#score = model.evaluate(x_test, y_test, verbose=0)
