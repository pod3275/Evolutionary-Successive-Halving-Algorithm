# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:30:58 2019

@author: lawle
"""
import os
from random import uniform
import numpy as np
import time
from math import ceil, floor, log
import concurrent.futures

import tensorflow as tf
from tensorflow.python.client import device_lib

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K

R = 54
etha = 3
n_search_times = 3
n_models_one_gpu = 11

f = open('HB_CIFAR10_simple_hadd_%sepochs.txt' % R, 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pbounds = {"learning_rate": (1e-2, 1e-5), 
           "first_block_filter": (0, 3.999999),
           "second_block_filter": (0, 3.999999),
           "dense_layers": (1, 4.999999),
           "dense_nodes": (32, 512.999999),
           "dense_act": (0, 2.999999),
           "dropout": (0.2, 1.0),
           "batch_size": (0, 2.999999)   }

global total_epochs
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

def get_random_config():
    # hparams = ['lr', 'fb_filter', 'sb_filter', 'd_layers', 'd_nodes', 'd_act', 'dropout', 'b_size'] 
    fb_list = [16,32,64,128]
    sb_list = [16,32,64,128]
    act_list = ['relu','sigmoid','tanh']
    b_size_list = [64,128,256]
       
    lr = uniform(pbounds["learning_rate"][0], pbounds["learning_rate"][1])
    
    fb_f_v = int(uniform(pbounds["first_block_filter"][0], pbounds["first_block_filter"][1]))
    sb_f_v = int(uniform(pbounds["second_block_filter"][0], pbounds["second_block_filter"][1]))
    
    d_layers = int(uniform(pbounds["dense_layers"][0], pbounds["dense_layers"][1]))
    d_nodes = int(uniform(pbounds["dense_nodes"][0], pbounds["dense_nodes"][1]))
    d_act_v = int(uniform(pbounds["dense_act"][0], pbounds["dense_act"][1]))
    
    dropout = uniform(pbounds["dropout"][0], pbounds["dropout"][1])
    batch_v = int(uniform(pbounds["batch_size"][0], pbounds["batch_size"][1]))
    
    fb_f = fb_list[fb_f_v]
    sb_f = sb_list[sb_f_v]
    d_act = act_list[d_act_v]
    b_size = b_size_list[batch_v]
                
    return [lr, fb_f, sb_f, d_layers, d_nodes, d_act, dropout, b_size]

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
    #params= ['gpu', 'lr', 'fb_filter', 'sb_filter', 
    #         'd_layers', 'd_nodes', 'd_act', 'dropout', 'b_size', 'weights'] : 10
    if len(params) > len(pbounds):
        with tf.Session(graph=tf.Graph()) as sess:
            K.set_session(sess)
            with tf.device(params[0]):
                print("model compile with %s start" % params[0])
                model = build_CNN(hparams=params[1:len(pbounds)])
                
                if len(params) == len(pbounds)+2:
                    model.set_weights(params[-1])
                #model에 params[5] 복사
                print("model compile with %s completed and fit start" % params[0])
                global epochs, init_epochs
                history = model.fit(x=train_data[0],
                                    y=train_data[1],
                                    epochs=epochs,
                                    batch_size=params[len(pbounds)],
                                    validation_data=validation_data,
                                    initial_epoch=init_epochs, verbose=0)
    
                print("model fit with %s completed" % params[0])
                accuracy = history.history['val_acc'][-1]
                weights = model.get_weights()
    
                return -accuracy, weights


def parallel_training(population, pop_model=None):
    result_pop_acc = []
    result_pop_model = []
    pop_size = len(population)
    
    global epochs, init_epochs
    
    for i in range(ceil(pop_size/num_gpus)):
        parallel_param = []
        
        if (i+1)*num_gpus > pop_size:
            sub_loop = pop_size - i*num_gpus
        else:
            sub_loop = num_gpus
            
        for j in range(sub_loop):
            pp_sub = []
            pp_sub.append(gpus[j])
            
            for c in range(len(pbounds)):
                pp_sub.append(population[i*num_gpus+j][c])
            
            if pop_model is not None:
                pp_sub.append(pop_model[i*num_gpus+j])
                
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
            result_pop_acc.append(results[m][0])
            result_pop_model.append(results[m][1])
    
    return result_pop_acc, result_pop_model


def SuccessiveHalving(s, n, r):
    population = []
    print("\n=============================================")
    f.write("\n=============================================\n")
    print("\nStart %sth bracket: n = %s, r = %s" % (s, n, r))
    f.write("\nStart %sth bracket: n = %s, r = %s\n" % (s, n, r))
    
    start_time1 = time.time()
    for i in range(n):
        population.append(get_random_config())

    global epochs, init_epochs
    
    epochs = 0
    for i in range(s+1):
        start_time = time.time()
        ri = r * etha**(i)
        if i == s:
            ri = R
            
        print("\n%sth bracket, ni = %s, ri = %s training" % (s, len(population), ri))
        f.write("\n%sth bracket, ni = %s, ri = %s training\n" % (s, len(population), ri))
        init_epochs = epochs
        epochs = ri
        #if i==0:
        pop_acc, pop_model = parallel_training(population)
        #else:
            #pop_acc, pop_model = parallel_training(population, pop_model)
            
        if i < s:
            sort = np.argsort(pop_acc)[:int(len(population)/etha)]
            
            new_population, new_pop_acc, new_pop_model = [], [], []
            
            for j in sort:
                new_population.append(population[j])
                new_pop_acc.append(pop_acc[j])
                new_pop_model.append(pop_model[j])
            
            population, pop_acc, pop_model = new_population, new_pop_acc, new_pop_model
            del(new_population, new_pop_acc, new_pop_model)
            
        f.write("Total spend epochs: %s\n" % total_epochs)
        f.write("Best accuracy: %s\n" %-min(pop_acc))
        f.write("Best param set: %s\n" % (population[np.argmin(pop_acc)]))
        
        end_time = time.time()
        #f.write("Time: %s, consuming: %s seconds\n" 
        #        % (time.strftime("%X",time.localtime()), round(end_time-start_time, 4)))
    
    end_time1 = time.time()
    f.write("\nTotal Bracket %s consuming: %s seconds\n" 
            % (s, round(end_time1-start_time1, 4)))
    
    f.write("Total spend epochs: %s\n" % total_epochs)
    f.write("Best accuracy: %s\n" %-min(pop_acc))
    f.write("Best param set: %s\n" % (population[np.argmin(pop_acc)]))
    return population, pop_acc, ("Bracket %s" %s)


def print_population(population, pop_acc, pop_from):
    for i in range(len(population)):
        f.write("\n%sth population\n" % (i+1))
        f.write("pop: %s\n" % population[i])
        f.write("pop_from: %s\n" % pop_from[i])
        f.write("pop_acc: %s\n" % -pop_acc[i])
        
    f.write("\nBest accuracy: %s\n" %-min(pop_acc))
    f.write("Best param set: %s\n" % (population[np.argmin(pop_acc)]))
    f.write("Best param from: %s\n" % (pop_from[np.argmin(pop_acc)]))
    f.write("Total spend epochs: %s\n" % total_epochs)


if __name__ == "__main__":
    total_start_time = time.strftime("%X",time.localtime())
    f.write("Start time: %s\n" % (total_start_time))
    f.write("Hyperband with R=%s, etha=%s\n" % (R, etha))
    
    smax = floor(log(R,etha))
    f.write("Number of brackets: %s\n" % (smax+1))
    
    result_pop = []
    result_pop_acc = []
    result_pop_from = []
    
    for i in range(smax+1):
        s = smax-i
        n = floor((smax+1)/(s+1)) * etha**s * n_search_times
        r = int(R * etha**(-s))
        
        b_pop, b_pop_acc, b_pop_from = SuccessiveHalving(s, n, r)
        if len(b_pop_acc) > 1:
            for j in range(len(b_pop_acc)):
                result_pop.append(b_pop[j])
                result_pop_acc.append(b_pop_acc[j])
                result_pop_from.append(b_pop_from)
        else:
            result_pop.append(b_pop[0])
            result_pop_acc.append(b_pop_acc[0])
            result_pop_from.append(b_pop_from)
    
    f.write("\n=============================================\n")
    f.write("==============Results==============\n")
    print_population(result_pop, result_pop_acc, result_pop_from)
        
    total_end_time = time.strftime("%X",time.localtime())
    f.write("\nEnd time: " + str(total_end_time))
    f.write("\nTotal spend epochs: %s\n" % total_epochs)
    
    f.close()
