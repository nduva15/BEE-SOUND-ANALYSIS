import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling1D
from keras.layers import Conv2D , MaxPooling2D, Conv1D, MaxPool1D, Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from datetime import datetime 
from keras.layers.normalization import BatchNormalization 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
#import my_tools as mt

#-------------2D - CNN-----------------------------#
def make_model_CNN(X_shape_1, X_shape_2, n_outputs):
    model=Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(X_shape_1, X_shape_2, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
#    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
#    model.add(LeakyReLU(alpha=0.1))
#    model.add(MaxPooling2D((2, 2), padding='same'))
#    model.add(Dropout(0.25))
#    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
#    model.add(LeakyReLU(alpha=0.1))
#    model.add(MaxPooling2D((2, 2), padding='same'))
#    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.25))
    model.add(Dense(32 , activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

#-------------1D - CNN-----------------------------#

def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding="same")(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding="same")(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=2)(x)


def make_model_1DCNN(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    #x = residual_block(x, 128, 3)

    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x); ## added by DF
    x = Dropout(0.25)(x) ## added by DF
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x); ## added by DF
    x = Dropout(0.5)(x)  ## added by DF
    x = Dense(16 , activation='relu')(x)  ## added by DF
    x = BatchNormalization()(x); ## added by DF
    x = Dropout(0.5)(x)  ## added by DF
    
    outputs = Dense(num_classes, activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=outputs)


def train_evaluate_CNN(X_train, Y_train, X_test, Y_test, n_outputs, num_batch_size, num_epochs, class_names, 
                   target_names):
    print('Training...')
    model = make_model_CNN(X_train.shape[1], X_train.shape[2], n_outputs )
    print(X_train.shape[0], X_train.shape[1])
    X_train = X_train.reshape(-1,  X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2],  1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    le = LabelEncoder()
    Y_train = to_categorical(le.fit_transform(Y_train)) 
    Y_test = to_categorical(le.fit_transform(Y_test)) 
    #Calculate pre-training accuracy 
    score = model.evaluate(X_test, Y_test, verbose=1)
    accuracy = 100*score[1]
    print("Predicted accuracy: ", accuracy)
    #Training the network
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    model_save_filename = "model.h5"

    earlystopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
    mdlcheckpoint_cb = ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
    )
    start = datetime.now()
    model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs,
                    validation_data=(X_test, Y_test), verbose=1, callbacks = [earlystopping_cb, mdlcheckpoint_cb])
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    # Evaluating the model on the training and testing set
    score1 = model.evaluate(X_train, Y_train, verbose=1)
    print("Training Accuracy: ", score1[1])
    score = model.evaluate(X_test, Y_test, verbose=1)
    print("Testing Accuracy: ", score[1])
     #predicting
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(np.round(Y_pred), axis=1)
    rounded_predictions = model.predict_classes(X_test, batch_size=128, verbose=0)
    print(rounded_predictions[1])
    rounded_labels=np.argmax(Y_test, axis=1)
    print(rounded_labels[1])
    #Confusion matrix
    cnf_matrix = confusion_matrix(rounded_labels, rounded_predictions)
    np.set_printoptions(precision=2)
    
    mt.plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix:')
    plt.show()
    print ('\nClasification report for fold:\n', 
           classification_report(rounded_labels, rounded_predictions, target_names=target_names ))
    return rounded_predictions, rounded_labels



