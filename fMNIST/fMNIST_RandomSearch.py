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

from hyperopt import hp, STATUS_OK
from hyperopt.pyll.stochastic import sample

from keras.datasets import fashion_mnist
from sklearn import metrics
from sklearn.svm import SVC
from keras import backend as K

init_pop_size = 15
etha = 3
one_budget = 1000
max_budgets = 27

f = open('Random_fMNIST_%spop_%sbudgets.txt' %(init_pop_size, max_budgets), 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

space = {
    'C': hp.uniform('C', -3, 2),
    'gamma': hp.uniform('gamma', -4, -1)
}

global total_budgets
total_budgets = 0

num_parallel = 8

def get_data():
    img_rows = 28
    img_cols = 28
        
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows*img_cols*1)
        x_test = x_test.reshape(x_test.shape[0], img_rows*img_cols*1)

    x_train = x_train.astype('float32')
    x_train /= 255
    
    x_test = x_test.astype('float32')
    x_test /= 255
    
    train_data = (x_train, y_train)
    validation_data = (x_test, y_test)
    
    return train_data, validation_data

train_data, validation_data = get_data()



def get_accuracy(params):
    C = 10 ** params['C']
    gamma = 10 ** params['gamma']
    
    classifier = SVC(C=C, gamma=gamma, random_state=1234)
    
    print("Fitting SVM")
    classifier.fit(train_data[0][:max_budgets*one_budget], train_data[1][:max_budgets*one_budget])
    
    print("Get accuracy of SVM")
    accuracy = metrics.accuracy_score(validation_data[1], 
                                      classifier.predict(validation_data[0]))
        
    return {'loss': -accuracy, 'params': params, 'status': STATUS_OK}



def parallel_training(population):
    result_pop_acc = []
    pop_size = len(population)
        
    for i in range(ceil(pop_size/num_parallel)):
        parallel_param = []
        
        if (i+1)*num_parallel > pop_size:
            sub_loop = pop_size - i*num_parallel
        else:
            sub_loop = num_parallel
            
        for j in range(sub_loop):                
            parallel_param.append(population[i*num_parallel+j])
            
        print("\nPopulation %sth to %sth parallel training" 
              % (i*num_parallel+1, min((i+1)*num_parallel, pop_size)))
        
        print("Training from budgets %s\n" % max_budgets)

        with concurrent.futures.ThreadPoolExecutor(len(parallel_param)) as executor:
            results = [x for x in executor.map(get_accuracy, parallel_param)]
        
        global total_budgets        
        total_budgets = total_budgets + max_budgets*len(parallel_param)
        
        for m in range(len(results)):
            result_pop_acc.append(results[m]['loss'])
    
    return result_pop_acc



def print_population(population, pop_acc):
    for i in range(len(population)):
        f.write("\n%sth population\n" % (i+1))
        f.write("pop: %s\n" % population[i])
        f.write("pop_acc: %s\n" % -pop_acc[i])
        
    f.write("\nBest accuracy: %s\n" %-min(pop_acc))
    f.write("Best param set: %s\n" % (population[np.argmin(pop_acc)]))
    f.write("Total spend budgets: %s\n" % total_budgets)
    f.write("Now time: %s\n" % time.strftime("%X",time.localtime()))



if __name__ == "__main__":
    total_start_time = time.strftime("%X",time.localtime())
    f.write("Start time: %s\n" % (total_start_time))
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
    f.write("\nRandom Sampling %s sets with %s budgets Training time: %s sec\n" 
            %(pop_size, max_budgets, round(end_time-start_time, 4)))
    print_population(population, pop_acc)
    
    total_end_time = time.strftime("%X",time.localtime())
    f.write("\nEnd time: " + str(total_end_time))
    f.write("\nTotal spend budgets: %s\n" % total_budgets)
            
    f.close()
    