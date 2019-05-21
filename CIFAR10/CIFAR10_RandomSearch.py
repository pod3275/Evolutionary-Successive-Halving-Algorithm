# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:30:58 2019

@author: lawle
"""
import os
import numpy as np
import time
from math import ceil
import concurrent.futures

import tensorflow as tf
from tensorflow.python.client import device_lib

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K

init_pop_size = 13
max_epochs = 54
n_models_one_gpu = 4

f = open('Random_CIFAR10_simple_hadd_%spop_%sepochs.txt' %(init_pop_size, max_epochs), 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

total_epochs = 0


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

gpus = get_available_gpus()
gpus = gpus*n_models_one_gpu
num_gpus = len(gpus)



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




def build_CNN(params):
    img_size = 32
    num_classes = 10
     
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
    
    return model



def get_accuracy(params):
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)
        with tf.device(params['gpu']):
            print("model compile with %s start" % params['gpu'])
            model = build_CNN(params)
            
            #model에 params[5] 복사
            print("model compile with %s completed and fit start" % params['gpu'])
            history = model.fit(x=train_data[0],
                                y=train_data[1],
                                epochs=epochs,
                                batch_size=params['batch_size'],
                                validation_data=validation_data,
                                initial_epoch=init_epochs, verbose=0)

            print("model fit with %s completed" % params['gpu'])
            accuracy = history.history['val_acc'][-1]

            return -accuracy



def parallel_training(population):
    result_pop_acc = []
    for i in range(ceil(pop_size/num_gpus)):
        parallel_param = []
        
        if (i+1)*num_gpus > pop_size:
            sub_loop = pop_size - i*num_gpus
        else:
            sub_loop = num_gpus
            
        for j in range(sub_loop):
            pp_sub = {}
            pp_sub['gpu'] = gpus[j]
            
            for c in population[i*num_gpus+j]:
                pp_sub[c] = population[i*num_gpus+j][c]
                            
            parallel_param.append(pp_sub)
            
        print("\nPopulation %sth to %sth parallel training" 
              % (i*num_gpus+1, min((i+1)*num_gpus, pop_size)))
        
        print("Training from epoch %s to %s\n" %(init_epochs, epochs))
        
        with concurrent.futures.ThreadPoolExecutor(len(parallel_param)) as executor:
            results = [x for x in executor.map(get_accuracy, parallel_param)]
        K.clear_session()
        
        global total_epochs
        total_epochs = total_epochs + (epochs-init_epochs)*len(parallel_param)
        
        for m in range(len(results)):
            result_pop_acc.append(results[m])
    
    return result_pop_acc




def print_population(population, pop_acc):
    for i in range(len(population)):
        f.write("\n%sth population\n" % (i+1))
        f.write("pop: %s\n" % population[i])
        f.write("pop_acc: %s\n" % -pop_acc[i])
        
    f.write("\nBest accuracy: %s\n" %-min(pop_acc))
    f.write("Best param set: %s\n" % (population[np.argmin(pop_acc)]))
    f.write("Total spend epochs: %s\n" % total_epochs)
    f.write("Time: %s\n" % (time.strftime("%X",time.localtime())))




if __name__ == "__main__":
    total_start_time = time.strftime("%X",time.localtime())
    f.write("Start time: %s\n" % (total_start_time))
    epochs=max_epochs
    init_epochs=0
    population = []
    pop_size = init_pop_size
    
    print("###### RANDOM SAMPLING Training ######")
    start_time = time.time()
    # random initialize
    for i in range(pop_size):
        population.append(sample(space))
   
    pop_acc = parallel_training(population)

    end_time = time.time()
    f.write("============================================================")
    f.write("\nRandom Sampling %s sets with %s epochs Training time: %s sec\n" 
            %(pop_size, epochs, round(end_time-start_time, 4)))
    print_population(population, pop_acc)
    
    
    total_end_time = time.strftime("%X",time.localtime())
    f.write("\nEnd time: " + str(total_end_time))
    f.write("\nTotal spend epochs: %s\n" % total_epochs)
            
    f.close()
    