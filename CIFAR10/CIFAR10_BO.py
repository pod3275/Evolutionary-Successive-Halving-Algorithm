# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:45:44 2019

@author: lawle
"""

import numpy as np
import time
import os
import tensorflow as tf
from skopt import Optimizer
from skopt.space import Real, Categorical, Integer

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K

epochs = 54
n_observe = 37

f = open('BO_CIFAR10_simple_hadd_%sobserve_%sepochs.txt' % (n_observe,epochs), 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dim_learning_rate = Real(low=1e-5, high=1e-2, prior='uniform',name='learning_rate')
dim_first_block_filter = Categorical(categories=[16,32,64,128], name='first_block_filter')
dim_second_block_filter = Categorical(categories=[16,32,64,128], name='second_block_filter')
dim_dense_layers = Integer(low=1, high=4, name='dense_layers')
dim_dense_nodes = Integer(low=32, high=512, name='dense_nodes')
dim_dense_act = Categorical(categories=['relu','sigmoid','tanh'], name='dense_act')
dim_dropout = Real(low=0.2, high=1.0, prior='uniform',name='dropout')
dim_batch_size = Categorical(categories=[64,128,256], name='batch_size')

dimensions = [dim_learning_rate, dim_first_block_filter, dim_second_block_filter, 
              dim_dense_layers, dim_dense_nodes, dim_dense_act, dim_dropout, dim_batch_size]

total_epochs = 0

def get_data():
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, 10)
    
    train_data = (x_train, y_train)
    validation_data = (x_test, y_test)
    
    return train_data, validation_data

train_data, validation_data = get_data()

# hparams = ['lr', 'fb_filter', 'sb_filter', 'd_layers', 'd_nodes', 'd_act', 'dropout', 'b_size']
def build_CNN(hparams):
    img_size = 32
    num_classes = 10
     
    model = Sequential()
    model.add(Conv2D(filters=hparams[1], kernel_size=(3, 3), padding='same',
                     input_shape=(img_size,img_size,3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(filters=hparams[1], kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hparams[6]))
    
    model.add(Conv2D(filters=hparams[2], kernel_size=(3, 3), padding='same',
                     kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(filters=hparams[2], kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hparams[6]))
    
    model.add(Flatten())
    for i in range(hparams[3]):
        model.add(Dense(hparams[4], activation=hparams[5])) #1
        
    model.add(Dropout(hparams[6]))
    model.add(Dense(num_classes, activation='softmax'))
           
    optimizer = Adam(lr=hparams[0])
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def get_accuracy(params):
    model = build_CNN(hparams=params[0:0+len(dimensions)-1])
   
    # Use Keras to train the model.
    print("Training from epoch %s to %s\n" %(init_epochs, epochs))
    history = model.fit(x=train_data[0],
                        y=train_data[1],
                        epochs=epochs,
                        batch_size=params[len(dimensions)-1],
                        validation_data=validation_data, 
                        initial_epoch=init_epochs)
    
    global total_epochs
    total_epochs = total_epochs + (epochs-init_epochs)
    
    accuracy = history.history['val_acc'][-1]
    
    return -accuracy


if __name__ == "__main__":
    total_start_time = time.strftime("%X",time.localtime())
    f.write("Start time: %s\n" % (total_start_time))
    init_epochs=0
    population = []
    pop_acc = []
            
    print("\n###### BAYESIAN OPTIMIZATION Training %s observes %s epochs ######" % (n_observe,epochs))
    f.write("###### BAYESIAN OPTIMIZATION Training %s observes %s epochs ######\n" % (n_observe,epochs))
    # another 5 with BO
    
    opt = Optimizer(
            dimensions=dimensions,
            acq_func="EI",
            n_initial_points=5,
            random_state=1234
    )
        
    for i in range(n_observe):
        next_set = opt.ask()
        print("\nBayesian Optimization %sth" %(i+1))
        print("observer: %s\n" % next_set)
        start_time = time.time()
        next_acc = get_accuracy(next_set)
        end_time = time.time()
        f.write("\n%sth observation\n" % (i+1))
        f.write("observer: %s\n" % next_set)
        f.write("pop_acc: %s\n" % -next_acc)
        #f.write("Time: %s, consuming: %s seconds\n" 
        #        % (time.strftime("%X",time.localtime()), round(end_time-start_time, 4)))
        f.write("Total spend epochs: %s\n" % total_epochs)
        
        opt.tell(next_set, next_acc)
        population.append(next_set)
        pop_acc.append(next_acc)
        
    f.write("============================================================")
    f.write("\nBest accuracy: %s\n" %-min(pop_acc))
    f.write("Best param set: %s\n" % (population[np.argmin(pop_acc)]))
         
    total_end_time = time.strftime("%X",time.localtime())
    f.write("\nEnd time: " + str(total_end_time))
    f.write("\nTotal spend epochs: %s\n" % total_epochs)
    f.close()
    