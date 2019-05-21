# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:51:38 2019

@author: lawle
"""

import os
import csv
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample

import tensorflow as tf

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam

max_budget = 54
n_iterations = 37

f = open('TPE_CIFAR10_simple_hadd_%sobserve_%sepochs.txt' %(n_iterations, max_budget), 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the search space
space = {
    'learning_rate': hp.uniform('learning_rate', 1e-5, 1e-2),
    'first_block_filter': hp.choice('first_block_filter', [16,32,64,128]),
    'second_block_filter': hp.choice('second_block_filter', [16,32,64,128]),
    'dense_layers': hp.quniform('dense_layers', 1, 4, 1),
    'dense_nodes': hp.quniform('dense_nodes', 32, 512, 1),
    'dense_act': hp.choice('dense_act', ['relu','sigmoid','tanh']),
    'dropout': hp.uniform('dropout', 0.2, 1.0),
    'batch_size': hp.choice('batch_size', [64,128,256])
}

cat_space = {
        'batch_size' : [64,128,256],
        'dense_act' : ['relu','sigmoid','tanh'],
        'first_block_filter' : [16,32,64,128],
        'second_block_filter' : [16,32,64,128]
}

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

def get_accuracy(params):
    img_size = 32
    num_classes = 10
    
    # Keep track of evals
    global ITERATION
    ITERATION += 1
    
    for parameter_name in ['dense_layers', 'dense_nodes']:
        params[parameter_name] = int(params[parameter_name])
        
    model = Sequential()
    model.add(Conv2D(filters=params['first_block_filter'], kernel_size=(3, 3),
                     padding='same', activation='relu', input_shape=(img_size,img_size,3),
                     kernel_initializer='he_normal'))
    model.add(Conv2D(filters=params['first_block_filter'], kernel_size=(3, 3),
                     activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout']))
    
    model.add(Conv2D(filters=params['second_block_filter'], kernel_size=(3, 3), 
                     activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(filters=params['second_block_filter'], kernel_size=(3, 3),
                     activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout']))
    
    model.add(Flatten())
    for i in range(params['dense_layers']):
        model.add(Dense(params['dense_nodes'], activation=params['dense_act'])) #1
        
    model.add(Dropout(params['dropout']))
    model.add(Dense(num_classes, activation='softmax'))
           
    optimizer = Adam(lr=params['learning_rate'])
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #------------------------------------------
    start = timer() 
    
    history = model.fit(x=train_data[0],
                        y=train_data[1],
                        epochs=max_budget,
                        batch_size=params['batch_size'],
                        validation_data=validation_data,
                        verbose=0)
    run_time = timer() - start
    
    global TOTAL_EPOCHS
    TOTAL_EPOCHS = TOTAL_EPOCHS + max_budget
    
    accuracy = history.history['val_acc'][-1]
    loss = -accuracy

    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([accuracy, params, ITERATION, run_time])
    
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


if __name__ == "__main__":
    # optimization algorithm
    tpe_algorithm = tpe.suggest
    
    # Keep track of results
    bayes_trials = Trials()
    
    # File to save first results
    out_file = ('TPE_CIFAR10_simple_hadd_%sobserve_%sepochs.csv' %(n_iterations, max_budget))
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)
    
    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration', 'train_time'])
    of_connection.close()
    
    # Global variable
    global ITERATION
    global TOTAL_EPOCHS
    
    ITERATION = 0
    TOTAL_EPOCHS=0
    
    # Run optimization
    best = fmin(fn = get_accuracy, space = space, algo = tpe_algorithm, 
                max_evals = n_iterations, trials = bayes_trials, rstate = np.random.RandomState(50))

    # Sort the trials with lowest loss (highest AUC) first
    bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])

    results = pd.read_csv('TPE_CIFAR10_simple_hadd_%sobserve_%sepochs.csv' %(n_iterations, max_budget))
    
    # Sort with best scores on top and reset index for slicing
    results.sort_values('loss', ascending = False, inplace = True)
    results.reset_index(inplace = True, drop = True)

    f.write("Best parameter set : %s" % results.loc[0, 'params'])
    f.write("\n\nBest accuracy : %s" % results.loc[0, 'loss'])
    f.write("\n\nBest set is %sth iteration" % results.loc[0, 'iteration'])
    f.write("\n\n\nTotal epochs : %s" % TOTAL_EPOCHS)
    
    f.close()
