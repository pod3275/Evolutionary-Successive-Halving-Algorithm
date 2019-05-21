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

from keras.datasets import fashion_mnist
from sklearn import metrics
from sklearn.svm import SVC

from keras import backend as K

# 1R = 1000 data
one_budget = 1000
R = 27
etha = 3
n_search_times = 2

f = open('HB_fMNIST_%sbudgets_%stimes.txt' % (R, n_search_times), 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pbounds = {"C": (-3, 2), "gamma": (-4, -1)}

global total_budgets
total_budgets = 0

num_parallel = 7

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


def get_random_config():
    C_v = uniform(pbounds["C"][0], pbounds["C"][1])
    gamma_v = uniform(pbounds["gamma"][0], pbounds["gamma"][1])
    
    C = 10**C_v
    gamma = 10 ** gamma_v
                
    return [C, gamma]


def get_accuracy(params):
    classifier = SVC(C=params[0], gamma=params[1], random_state=1234)
    
    # Use Keras to train the model.
    print("Fitting SVM")
    classifier.fit(train_data[0][:budgets*one_budget], train_data[1][:budgets*one_budget])
    
    print("Get accuracy of SVM")
    accuracy = metrics.accuracy_score(validation_data[1], 
                                      classifier.predict(validation_data[0]))
    
    return -accuracy


def parallel_training(population):
    result_pop_acc = []
    pop_size = len(population)
    
    global budgets
    
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
        
        print("Training from budgets %s\n" % budgets)

        with concurrent.futures.ThreadPoolExecutor(len(parallel_param)) as executor:
            results = [x for x in executor.map(get_accuracy, parallel_param)]
        
        global total_budgets        
        total_budgets = total_budgets + budgets*len(parallel_param)
        
        for m in range(len(results)):
            result_pop_acc.append(results[m])
    
    return result_pop_acc


def SuccessiveHalving(s, n, r):
    population = []
    print("\n=============================================")
    f.write("\n=============================================\n")
    print("\nStart %sth bracket: n = %s, r = %s" % (s, n, r))
    f.write("\nStart %sth bracket: n = %s, r = %s\n" % (s, n, r))
    
    start_time1 = time.time()
    for i in range(n):
        population.append(get_random_config())

    global budgets
    budgets = 0
    
    for i in range(s+1):
        ri = r * etha**(i)
        if i == s:
            ri = R
            
        start_time = time.time()
        print("\n%sth bracket, ni = %s, ri = %s training" % (s, len(population), ri))
        f.write("\n%sth bracket, ni = %s, ri = %s training\n" % (s, len(population), ri))
        budgets = ri
        
        pop_acc = parallel_training(population)
            
        if i < s:
            sort = np.argsort(pop_acc)[:int(len(population)/etha)]
            
            new_population, new_pop_acc = [], []
            
            for j in sort:
                new_population.append(population[j])
                new_pop_acc.append(pop_acc[j])
            
            population, pop_acc = new_population, new_pop_acc
            del(new_population, new_pop_acc)
            
        end_time = time.time()
        f.write("Total spend budgets: %s\n" % total_budgets)
        f.write("Best accuracy: %s\n" %-min(pop_acc))
        f.write("Best param set: %s\n" % (population[np.argmin(pop_acc)]))
        f.write("Time: %s, consuming: %s seconds\n" 
                % (time.strftime("%X",time.localtime()), round(end_time-start_time, 4)))
    
    end_time1 = time.time()
    f.write("\nBracket %s Total consuming time: %s seconds\n" 
            % (s, round(end_time1-start_time1, 4)))
    
    f.write("Total spend budgets: %s\n" % total_budgets)
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
    f.write("Total spend budgets: %s\n" % total_budgets)


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
    
    f.close()
