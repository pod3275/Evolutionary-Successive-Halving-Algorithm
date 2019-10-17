# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:15:03 2019

@author: lawle
"""

import os
import numpy as np
import time
import csv
from math import ceil, floor, log
import concurrent.futures

import tensorflow as tf
from tensorflow.python.client import device_lib

import hyperopt
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample

import keras
import math
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from functools import partial

init_pop_size = 525
etha = 3
max_epochs = 81
first_epoch = int(max_epochs / etha**floor(log(max_epochs, etha)))
bayes_n = 2
n_models_one_gpu = 7 # 높으면 OOM 발생
lambda_ = 0.3
early_stopping = True
early_stop_step = 13
pre_training = True
pre_train_step = 3

csv_f = open('ESHA_4blocks_TPE_CIFAR10_lambda_%s_es=%s_%s_pt=%s_%s_%sx_%spop_%sepochs.csv' %(lambda_, early_stopping, early_stop_step, pre_training, pre_train_step, bayes_n, init_pop_size, max_epochs), 'w', newline='')

f = csv.writer(csv_f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

space = {
    'learning_rate': hp.loguniform('learning_rate', math.log(1e-5), math.log(1e-2)),
    'first_block_filter': hp.quniform('first_block_filter', 16, 64, 1),
    'second_block_filter': hp.quniform('second_block_filter', 16, 64, 1),
    'third_block_filter': hp.quniform('third_block_filter', 16, 64, 1),
    'fourth_block_filter': hp.quniform('fourth_block_filter', 16, 64, 1),
    'dense_layers': hp.quniform('dense_layers', 1, 4, 1),
    'dense_nodes': hp.quniform('dense_nodes', 8, 256, 1),
    'dense_act': hp.choice('dense_act', ['relu','sigmoid','tanh']),
    'dropout': hp.uniform('dropout', 0.2, 0.8),
    'batch_size': hp.choice('batch_size', [64,128,256])
}


weight_dict = {}
shape_dict = {}

total_epochs = 0

cat_space = {
        'batch_size' : [64,128,256],
        'dense_act' : ['relu','sigmoid','tanh']
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
     
    for parameter_name in ['dense_layers', 'dense_nodes',
                           'first_block_filter', 'second_block_filter',
                           'third_block_filter', 'fourth_block_filter']:
        params[parameter_name] = int(params[parameter_name])
     
        
    model = Sequential()
    model.add(Conv2D(filters=params['first_block_filter'], kernel_size=(3, 3),
                     padding='same', activation='relu', input_shape=(img_size,img_size,3),
                     kernel_initializer='he_normal', name="conv1_1"))
    model.add(Conv2D(filters=params['first_block_filter'], kernel_size=(3, 3),
                     padding='same', activation='relu', kernel_initializer='he_normal', name="conv1_2"))
    model.add(Dropout(params['dropout']))
    
    
    model.add(Conv2D(filters=params['second_block_filter'], kernel_size=(3, 3), 
                     activation='relu', padding='same', kernel_initializer='he_normal', name="conv2_1"))
    model.add(Conv2D(filters=params['second_block_filter'], kernel_size=(3, 3),
                     activation='relu', padding='same', kernel_initializer='he_normal', name="conv2_2"))
    model.add(Dropout(params['dropout']))
    
    
    model.add(Conv2D(filters=params['third_block_filter'], kernel_size=(3, 3),
                     padding='same', activation='relu', kernel_initializer='he_normal', name="conv3_1"))
    model.add(Conv2D(filters=params['third_block_filter'], kernel_size=(3, 3),
                     padding='same', activation='relu', kernel_initializer='he_normal', name="conv3_2"))
    model.add(Dropout(params['dropout']))
    
    
    model.add(Conv2D(filters=params['fourth_block_filter'], kernel_size=(3, 3),
                     padding='same', activation='relu', kernel_initializer='he_normal', name="conv4_1"))
    model.add(Conv2D(filters=params['fourth_block_filter'], kernel_size=(3, 3),
                     padding='same', activation='relu', kernel_initializer='he_normal', name="conv4_2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout']))

    
    model.add(Flatten())
    
    for i in range(params['dense_layers']):
        model.add(Dense(params['dense_nodes'], activation=params['dense_act'], name="dense_"+str(i))) #1
        
    model.add(Dropout(params['dropout']))
    model.add(Dense(num_classes, activation='softmax', name="dense_last"))
           
    optimizer = Adam(lr=params['learning_rate'])
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model



def save_biggest_weight(loc = "./weights/", save = False):
    max_params = {
        'learning_rate': 1e-3,
        'first_block_filter': 64,
        'second_block_filter': 64,
        'third_block_filter': 64,
        'fourth_block_filter': 64,
        'dense_layers': 4,
        'dense_nodes': 256,
        'dense_act': 'relu',
        'dropout': 0.2,
        'batch_size': 128
    }
    
    model = build_CNN(max_params)
    
    early_stopping = EarlyStopping(monitor='val_acc', patience = early_stop_step)
    model_checkpoint = ModelCheckpoint(filepath='max_model.h5',
                                       monitor='val_acc',
                                       save_best_only=True)
    
    callbacks = [early_stopping, model_checkpoint]
    
    print("Pre-training Biggest Model")
    
    history = model.fit(x=train_data[0],
                        y=train_data[1],
                        epochs=pre_train_step,
                        batch_size=max_params['batch_size'],
                        validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=1)
    
    stop_index = np.argmax(history.history['val_acc'])
    valid_accuracy = history.history['val_acc'][stop_index]
    
    f.writerow(["pre-trained model results", len(history.history['acc']), valid_accuracy])
    
    model = load_model('max_model.h5')
    
    global total_epochs
    total_epochs = total_epochs + len(history.history['acc'])
    
    for layer in model.layers:
        if layer.get_weights() != []:
            weight_dict[layer.name] = layer.get_weights()[0]
            shape_dict[layer.name] = np.shape(layer.get_weights()[0])
            
            if save:
                if not os.path.exists(loc):
                    os.mkdir(loc)
                np.savetxt(loc + layer.name +".csv", layer.get_weights()[0].flatten(), delimiter=",")



def weight_initialize(model, loc = "./weights/", file_load = False):
    for layer in model.layers:
        if len(layer.get_weights()) == 2:
            l_shape = np.shape(layer.get_weights()[0])
            
            if file_load:
                a = np.loadtxt(loc + layer.name + ".csv", dtype=np.float32).reshape(shape_dict[layer.name])
            else:
                a = weight_dict[layer.name]
            
            if len(shape_dict[layer.name]) == 2:
                layer.set_weights([a[:l_shape[0],:l_shape[1]], np.zeros([l_shape[1]])])
            elif len(shape_dict[layer.name]) == 4:
                layer.set_weights([a[:l_shape[0],:l_shape[1], :l_shape[2], :l_shape[3]], np.zeros([l_shape[3]])])
    
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
                elif pre_training:
                    model = weight_initialize(model)
                    
                print("model compile with %s completed and fit start" % params['gpu'])
                
                if params['early_stop']:
                    print("Final: Early Stopping mode")
                    early_stopping = EarlyStopping(monitor='val_acc', patience=early_stop_step)
                    
                    model_checkpoint = ModelCheckpoint(filepath='CIFAR10earlystop_'+str(params['model_number']) + '.h5',
                                                       monitor='val_acc',
                                                       save_best_only=True)
                    
                    callbacks = [early_stopping, model_checkpoint]
                    
                    history = model.fit(x=train_data[0],
                                        y=train_data[1],
                                        epochs=epochs,
                                        batch_size=params['batch_size'],
                                        validation_data=validation_data,
                                        initial_epoch=init_epochs, 
                                        callbacks=callbacks,
                                        verbose=0)
                    
                    model2 = load_model('CIFAR10earlystop_'+str(params['model_number']) + '.h5')
                    
                    print("model fit with %s completed" % params['gpu'])
                    stop_index = np.argmax(history.history['val_acc'])
                    
                    train_accuracy = history.history['acc'][stop_index]
                    valid_accuracy = history.history['val_acc'][stop_index]
                    weights = model2.get_weights()
                    training_epochs = stop_index+1
                    
                else:
                    history = model.fit(x=train_data[0],
                                        y=train_data[1],
                                        epochs=epochs,
                                        batch_size=params['batch_size'],
                                        validation_data=validation_data,
                                        initial_epoch=init_epochs,
                                        verbose=0)
    
                    print("model fit with %s completed" % params['gpu'])
                    train_accuracy = history.history['acc'][-1]
                    valid_accuracy = history.history['val_acc'][-1]
                    weights = model.get_weights()
                    training_epochs = epochs-init_epochs
    
        return -train_accuracy, -valid_accuracy, weights, training_epochs
            
    else:
        with tf.Session(graph=tf.Graph()) as sess:
            K.set_session(sess)
            
            model = build_CNN(params)
            
            if pre_training:
                model = weight_initialize(model)
            
            # print("Training from epoch %s to %s\n" %(init_epochs, epochs))
            
            history = model.fit(x=train_data[0],
                                y=train_data[1],
                                epochs=epochs,
                                batch_size=params['batch_size'],
                                validation_data=validation_data, 
                                initial_epoch=init_epochs, verbose=0)
            
            global total_epochs
            total_epochs = total_epochs + (epochs-init_epochs)
            
            train_accuracy = history.history['acc'][-1]
            valid_accuracy = history.history['val_acc'][-1]
            weights = model.get_weights()
            
        return {'loss': -valid_accuracy, 'params': params, 'model' : weights, 'another_loss': -train_accuracy, 'status': STATUS_OK}



def get_accuracy_train(params):
    if 'gpu' not in params:
        with tf.Session(graph=tf.Graph()) as sess:
            K.set_session(sess)
            
            model = build_CNN(params)
            
            if pre_training:
                model = weight_initialize(model)
            
            # print("Training from epoch %s to %s\n" %(init_epochs, epochs))
            
            history = model.fit(x=train_data[0],
                                y=train_data[1],
                                epochs=epochs,
                                batch_size=params['batch_size'],
                                validation_data=validation_data, 
                                initial_epoch=init_epochs, verbose=0)
            
            global total_epochs
            total_epochs = total_epochs + (epochs-init_epochs)
            
            train_accuracy = history.history['acc'][-1]
            valid_accuracy = history.history['val_acc'][-1]
            weights = model.get_weights()
            
        return {'loss': -train_accuracy, 'params': params, 'model' : weights, 'another_loss': -valid_accuracy, 'status': STATUS_OK}



def parallel_training(population, pop_model=None, early_stop=False):
    result_pop_train_acc = []
    result_pop_valid_acc = []
    result_pop_model = []
    result_pop_last_epochs = []
    
    for i in range(ceil(pop_size/num_gpus)):
        parallel_param = []
        
        # param 개수 계산
        if (i+1)*num_gpus > pop_size:
            sub_loop = pop_size - i*num_gpus
        else:
            sub_loop = num_gpus
        
        # param 채우기
        for j in range(sub_loop):
            pp_sub = {}
            pp_sub['gpu'] = gpus[j]
            
            for c in population[i*num_gpus+j]:
                pp_sub[c] = population[i*num_gpus+j][c]
            
            if pop_model is not None:
                pp_sub['weights'] = pop_model[i*num_gpus+j]
                
            pp_sub['early_stop'] = early_stop
            pp_sub['model_number'] = j
            
            parallel_param.append(pp_sub)
            
        print("\nPopulation %sth to %sth parallel training" 
              % (i*num_gpus+1, min((i+1)*num_gpus, pop_size)))
        
        print("Training from epoch %s to %s\n" %(init_epochs, epochs))
        
        with concurrent.futures.ThreadPoolExecutor(len(parallel_param)) as executor:
            results = [x for x in executor.map(get_accuracy, parallel_param)]
        
        K.clear_session()
        
        global total_epochs
        for m in range(len(results)):
            result_pop_train_acc.append(results[m][0])
            result_pop_valid_acc.append(results[m][1])
            result_pop_model.append(results[m][2])
            total_epochs = total_epochs + results[m][3]
            if early_stop:
                result_pop_last_epochs.append(init_epochs + results[m][3])
    
    if early_stop:
        return result_pop_train_acc, result_pop_valid_acc, result_pop_model, result_pop_last_epochs
    else:
        return result_pop_train_acc, result_pop_valid_acc, result_pop_model



def print_population(population, pop_train_acc, pop_valid_acc, pop_from, verbose = 1):
    if verbose:
        for i in range(len(population)):
            f.writerow(["%sth" % (i+1), population[i], pop_from[i], -pop_train_acc[i], -pop_valid_acc[i]])
        
    f.writerow(["Best accuracy: %s" %-min(pop_valid_acc)])
    f.writerow(["Best", population[np.argmin(pop_valid_acc)], pop_from[np.argmin(pop_valid_acc)], 
                                   -pop_train_acc[np.argmin(pop_valid_acc)], 
                                   -pop_valid_acc[np.argmin(pop_valid_acc)]])
    f.writerow(["Total spend epochs: %s" % total_epochs])
#    f.writerow(["Total spend epochs: %s" % total_epochs, "Time: %s" % (time.strftime("%X",time.localtime()))])



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



def convert_results_to_trials(population, pop_acc, pop_another_acc):
    points = []
    for i in range(len(population)):
        temp = {}
        temp['loss'] = pop_acc[i]
        temp['params'] = population[i]
        temp['model'] = None
        temp['another_loss'] = pop_another_acc[i]
        temp['status'] = STATUS_OK
        
        points.append(temp)
    
    return points
   
    

def select_population(sort, population, pop_train_acc, pop_valid_acc, pop_model, pop_from):
    new_population, new_pop_train_acc, new_pop_valid_acc, new_pop_model, new_pop_from = [], [], [], [], []
        
    for i in sort:
        new_population.append(population[i])
        new_pop_train_acc.append(pop_train_acc[i])
        new_pop_valid_acc.append(pop_valid_acc[i])
        new_pop_model.append(pop_model[i])
        new_pop_from.append(pop_from[i])
    
    population, pop_train_acc, pop_valid_acc, pop_model, pop_from = new_population, new_pop_train_acc, new_pop_valid_acc, new_pop_model, new_pop_from
    del(new_population, new_pop_train_acc, new_pop_valid_acc, new_pop_model, new_pop_from)
    
    return population, pop_train_acc, pop_valid_acc, pop_model, pop_from

    

if __name__ == "__main__":
    total_start_time = time.time()
    f.writerow(["Start time: %s" % (time.strftime("%X", time.gmtime(total_start_time)))])
    f.writerow(['NUMBER', 'POPULATION', 'POP_FROM', 'TRAIN_ACC', 'VALID_ACC'])
    epochs=first_epoch
    init_epochs=0
    population = []
    pop_from = []
    pop_size = init_pop_size
    early_stop_ = False
    
    print("ESHA with R = %s, etha = %s, k = %s, bayes_n = %s" 
          % (max_epochs, etha, init_pop_size, bayes_n))
    print("============================================================")
    print("###### Training in 1st Population ######\n")
    start_time = time.time()
    
    if pre_training:
        save_biggest_weight()
    
    for i in range(pop_size):
        population.append(sample(space))
        pop_from.append("Random")
   
    pop_train_acc, pop_valid_acc, pop_model = parallel_training(population)

    end_time = time.time()
    f.writerow(["============================================================"])
    f.writerow(["Random Sampling %s sets with %s epochs Training time: %s sec" 
            %(pop_size, epochs, round(end_time-start_time, 4))])
    print_population(population, pop_train_acc, pop_valid_acc, pop_from, verbose=0)
    
    for s in range(ceil(log(max_epochs/first_epoch, etha))):
        print("\n============================================================")
        print("###### Expansion with Bayesian Optimization in %sth Population ######" % (s+1))
        print("###### Training %s epochs, %s sets ######\n" % (epochs, pop_size*bayes_n))
                
        # TPE with valid_acc
        print("(1) BO with valid_acc\n")
        tpe_algorithm = tpe.suggest
        
        bayes_trials = add_trials(convert_results_to_trials(population, pop_valid_acc, pop_train_acc))
        
        init_epochs=0
        
        start_time = time.time()
        best = fmin(fn = get_accuracy, space = space, algo = partial(tpe_algorithm, gamma=0.2, n_startup_jobs=0), 
                    max_evals = pop_size + int(pop_size * bayes_n/2), trials = bayes_trials, 
                    rstate = np.random.RandomState(1234))
        
        f.writerow(["============================================================"])
        f.writerow(["###### BAYESIAN OPTIMIZATION sampling %s sets, trainied with %s epochs ######" 
              % (pop_size * bayes_n, epochs)])
      
        cc=1
        for i in bayes_trials.results[pop_size:]:
            population.append(i['params'])
            pop_train_acc.append(i['another_loss'])
            pop_valid_acc.append(i['loss'])
            pop_model.append(i['model'])
            pop_from.append("Bayesian " + str(epochs) + '_valid')
            
            f.writerow(["%sth" % (cc), i['params'], "Bayesian %s_valid" % str(epochs), -i['another_loss'], -i['loss']])
            #f.writerow(["Time: %s\n" % (time.strftime("%X",time.localtime()))])
            
            cc+=1
            
        # TPE with train_acc
        print("(2) BO with train_acc\n")
        tpe_algorithm_t = tpe.suggest
        
        bayes_trials_t = add_trials(convert_results_to_trials(population, pop_train_acc, pop_valid_acc))
        
        init_epochs=0
        
        best = fmin(fn = get_accuracy_train, space = space, algo = partial(tpe_algorithm_t, gamma=0.2, n_startup_jobs=0), 
                    max_evals = pop_size + pop_size * bayes_n, trials = bayes_trials_t, 
                    rstate = np.random.RandomState(567))
        
        end_time = time.time()
      
        for i in bayes_trials_t.results[pop_size + int(pop_size * bayes_n/2):]:
            population.append(i['params'])
            pop_train_acc.append(i['loss'])
            pop_valid_acc.append(i['another_loss'])
            pop_model.append(i['model'])
            pop_from.append("Bayesian " + str(epochs) + '_train')
            
            f.writerow(["%sth" % (cc), i['params'], "Bayesian %s_train" % str(epochs), -i['loss'], -i['another_loss']])
            #f.writerow(["Time: %s\n" % (time.strftime("%X",time.localtime()))])
            
            cc+=1
        
        f.writerow(["BO Training time: %s sec" %round(end_time-start_time, 4)])
        f.writerow(["Total spend epochs: %s" % total_epochs])
        f.writerow(["============================================================"])
        
        print("\n============================================================")
        print("###### Selection %s sets in %sth Population ######" % (round(pop_size/etha), s+1))
        
        select_criteria = [(-pop_valid_acc[i])**2 + (1+pop_train_acc[i])*lambda_ for i in range(len(pop_valid_acc))]
        sort = np.argsort(select_criteria)[::-1][:round(pop_size/etha)]
        
        population, pop_train_acc, pop_valid_acc, pop_model, pop_from = select_population(sort,  
                                                                                          population, 
                                                                                          pop_train_acc, 
                                                                                          pop_valid_acc, 
                                                                                          pop_model, 
                                                                                          pop_from)
            
        pop_size = len(population)
        f.writerow(["After Selection, %s sets with %s epochs population results" % (pop_size, epochs)])
        print_population(population, pop_train_acc, pop_valid_acc, pop_from, verbose=1)
                   
        # Train epoch*etha
        init_epochs = epochs
        epochs = epochs*etha
        if s == ceil(log(max_epochs/first_epoch, etha))-1:
            epochs = max_epochs
            early_stop_ = early_stopping
            
        print("============================================================")
        print("###### Training in %sth Population to %s epochs ######\n" % (s+2, epochs))
        start_time = time.time()
        
        if early_stop_:
            new_pop_train_acc, new_pop_valid_acc, new_pop_model, pop_last_epochs = parallel_training(population, pop_model, early_stop=early_stop_)
        else:
            new_pop_train_acc, new_pop_valid_acc, new_pop_model = parallel_training(population, pop_model, early_stop=early_stop_)
                        
        pop_train_acc, pop_valid_acc, pop_model = new_pop_train_acc, new_pop_valid_acc, new_pop_model
        del(new_pop_train_acc, new_pop_valid_acc, new_pop_model)
        
        end_time = time.time()
        
        f.writerow(["============================================================"])
        f.writerow(["Population more training to %s epoch" % epochs])        
        f.writerow(["After More training to %s epochs in %s sets, population results" %(epochs, pop_size)])
        print_population(population, pop_train_acc, pop_valid_acc, pop_from, verbose=1)
        
        if early_stop_:
            f.writerow(["Last Training epochs"] + pop_last_epochs)
    
    total_end_time = time.time()
    f.writerow(["End time: %s" %(time.strftime("%X", time.gmtime(total_end_time)))])
    f.writerow(["Spending time: %s" %(time.strftime("%X", time.gmtime(total_end_time-total_start_time)))])
            
    csv_f.close()
    
