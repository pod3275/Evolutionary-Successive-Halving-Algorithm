 # -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:30:58 2019

@author: lawle
"""
import os
import numpy as np
import time
from math import ceil, floor, log
import concurrent.futures

import tensorflow as tf
from tensorflow.python.client import device_lib

import hyperopt
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K
from functools import partial

init_pop_size = 252
etha = 3
max_epochs = 54
first_epoch = int(max_epochs / etha**floor(log(max_epochs, etha)))
n_models_one_gpu = 12

f = open('ESHA_TPE_CIFAR10_simple_hadd_%spop_%sepochs.txt' %(init_pop_size, max_epochs), 'w')

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

cat_space = {
        'batch_size' : [64,128,256],
        'dense_act' : ['relu','sigmoid','tanh'],
        'first_block_filter' : [16,32,64,128],
        'second_block_filter' : [16,32,64,128]
}


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
    if 'gpu' in params:
        with tf.Session(graph=tf.Graph()) as sess:
            K.set_session(sess)
            with tf.device(params['gpu']):
                print("model compile with %s start" % params['gpu'])
                model = build_CNN(params)
                
                if 'weights' in params:
                    model.set_weights(params['weights'])
                    
                print("model compile with %s completed and fit start" % params['gpu'])
                history = model.fit(x=train_data[0],
                                    y=train_data[1],
                                    epochs=epochs,
                                    batch_size=params['batch_size'],
                                    validation_data=validation_data,
                                    initial_epoch=init_epochs, verbose=0)
    
                print("model fit with %s completed" % params['gpu'])
                accuracy = history.history['val_acc'][-1]
                weights = model.get_weights()
    
                return -accuracy, weights
            
    else:
        with tf.Session(graph=tf.Graph()) as sess:
            K.set_session(sess)
            
            model = build_CNN(params)
            
            print("Training from epoch %s to %s\n" %(init_epochs, epochs))
            
            history = model.fit(x=train_data[0],
                                y=train_data[1],
                                epochs=epochs,
                                batch_size=params['batch_size'],
                                validation_data=validation_data, 
                                initial_epoch=init_epochs, verbose=0)
            
            global total_epochs
            total_epochs = total_epochs + (epochs-init_epochs)
            
            accuracy = history.history['val_acc'][-1]
            weights = model.get_weights()
            
            return {'loss': -accuracy, 'params': params, 'model' : weights, 'status': STATUS_OK}



def parallel_training(population, pop_model=None):
    result_pop_acc = []
    result_pop_model = []
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
            
            if pop_model is not None:
                pp_sub['weights'] = pop_model[i*num_gpus+j]
                
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
    f.write("Time: %s\n" % (time.strftime("%X",time.localtime())))



def add_trials(points):
    test_trials = Trials()
    
    for tid, row in enumerate(points):
        vals = {}
        for key in sample(space).keys():
            if key in cat_space:
                vals[key] = [cat_space[key].index(row['params'][key])]
            else:
                vals[key] = [row['params'][key]]
                                
        
        hyperopt_trial = Trials().new_trial_docs(
            tids=[tid],
            specs=[None], 
            results=[row],
            miscs=[{'tid': tid, 
                    'cmd': ('domain_attachment', 'FMinIter_Domain'), 
                    'workdir': None,
                    'idxs': {**{key: [tid] for key in sample(space).keys()}}, 
                    'vals': vals
                    }]
           )

        hyperopt_trial[0]['state'] = hyperopt.JOB_STATE_DONE
        
        test_trials.insert_trial_docs(hyperopt_trial) 
        test_trials.refresh()
    
    return test_trials



def convert_results_to_trials(population, pop_acc):
    points = []
    for i in range(len(population)):
        temp = {}
        temp['loss'] = pop_acc[i]
        temp['params'] = population[i]
        temp['model'] = None
        temp['status'] = STATUS_OK
        
        points.append(temp)
    
    return points
   


if __name__ == "__main__":
    total_start_time = time.strftime("%X",time.localtime())
    f.write("Start time: %s\n" % (total_start_time))
    epochs=first_epoch
    init_epochs=0
    population = []
    pop_from = []
    pop_size = init_pop_size
    
    print("###### RANDOM SAMPLING Training ######")
    start_time = time.time()
    
    for i in range(pop_size):
        population.append(sample(space))
        pop_from.append("Random")
   
    pop_acc, pop_model = parallel_training(population)

    end_time = time.time()
    f.write("============================================================")
    f.write("\nRandom Sampling %s sets with %s epochs Training time: %s sec\n" 
            %(pop_size, epochs, round(end_time-start_time, 4)))
    print_population(population, pop_acc, pop_from)
    
    for s in range(int(log(max_epochs/first_epoch, etha))):
        print("\n###### BAYESIAN OPTIMIZATION Training %s phase, %s sets ######" 
              % (epochs, round(pop_size/etha)))
                
        tpe_algorithm = tpe.suggest
        
        bayes_trials = add_trials(convert_results_to_trials(population, pop_acc))
        
        init_epochs=0
        
        start_time = time.time()
        best = fmin(fn = get_accuracy, space = space, algo = partial(tpe_algorithm, gamma=0.2, n_startup_jobs=0), 
                    max_evals = pop_size + round(pop_size/etha), trials = bayes_trials, 
                    rstate = np.random.RandomState(50))
        
        end_time = time.time()
        f.write("============================================================")
        f.write("\n###### BAYESIAN OPTIMIZATION sampling %s sets, trainied with %s epochs ######\n" 
              % (round(pop_size/etha), epochs))
      
        cc=1
        for i in bayes_trials.results[pop_size:]:
            population.append(i['params'])
            pop_acc.append(i['loss'])
            pop_model.append(i['model'])
            pop_from.append("Bayesian " + str(epochs))
            
            f.write("\nBayesian Optimization %sth set with %s epochs\n" % (cc, epochs))
            f.write("pop: %s\n" % i['params'])
            f.write("pop_from: Bayesian %s\n" % str(epochs))
            f.write("pop_acc: %s\n" % -i['loss'])
            f.write("Time: %s\n" % (time.strftime("%X",time.localtime())))
            cc+=1
        
        f.write("BO Training time: %s sec" %round(end_time-start_time, 4))
        f.write("\nTotal spend epochs: %s" % total_epochs)
        f.write("\n============================================================")
        
        print("###### Remove and retain %s sets in population ######" % round(pop_size/etha))
        sort = np.argsort(pop_acc)[:round(pop_size/etha)]
        
        new_population, new_pop_acc, new_pop_model, new_pop_from = [], [], [], []
        
        for i in sort:
            new_population.append(population[i])
            new_pop_acc.append(pop_acc[i])
            new_pop_model.append(pop_model[i])
            new_pop_from.append(pop_from[i])
        
        population, pop_acc, pop_model, pop_from = new_population, new_pop_acc, new_pop_model, new_pop_from
        del(new_population, new_pop_acc, new_pop_model, new_pop_from)
        
        pop_size = len(population)
        f.write("\nAfter insert BO, %s sets with %s epochs population results\n" % (pop_size, epochs))
        print_population(population, pop_acc, pop_from)
                   
        # Train epoch*etha
        init_epochs = epochs
        epochs = epochs*etha
        if s == int(log(max_epochs/first_epoch, etha))-1:
            epochs = max_epochs
        
        print("###### Population More Training to %s epochs ######" % epochs)
        start_time = time.time()
        
        new_pop_acc, new_pop_model = parallel_training(population, pop_model)
                
        pop_acc, pop_model = new_pop_acc, new_pop_model
        del(new_pop_acc, new_pop_model)
        
        end_time = time.time()
        
        f.write("============================================================")
        #f.write("\nPopulation more training to %s epoch Training time: %s sec\n" 
        #        %(epochs, round(end_time-start_time, 4)))
        f.write("\nPopulation more training to %s epoch" % epochs)
        f.write("\nTotal spend epochs: %s\n" % total_epochs)
        
        f.write("After More training to %s epochs in %s sets, population results\n" %(epochs, pop_size))
        print_population(population, pop_acc, pop_from)
        
    
    total_end_time = time.strftime("%X",time.localtime())
    f.write("\nEnd time: " + str(total_end_time))
    f.write("\nTotal spend epochs: %s\n" % total_epochs)
            
    f.close()
    