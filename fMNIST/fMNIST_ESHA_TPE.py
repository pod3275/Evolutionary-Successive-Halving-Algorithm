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

import hyperopt
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample

from keras.datasets import fashion_mnist
from sklearn import metrics
from sklearn.svm import SVC
from keras import backend as K
from functools import partial

init_pop_size = 162
etha = 3
one_budget = 1000
max_budgets = 27
first_budget = int(max_budgets / etha**floor(log(max_budgets, etha)))

f = open('ESHA_TPE_fMNIST_%spop_%sbudgets.txt' %(init_pop_size, max_budgets), 'w')

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
    classifier.fit(train_data[0][:budgets*one_budget], train_data[1][:budgets*one_budget])
    
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
        
        print("Training from budgets %s\n" % budgets)

        with concurrent.futures.ThreadPoolExecutor(len(parallel_param)) as executor:
            results = [x for x in executor.map(get_accuracy, parallel_param)]
        
        global total_budgets        
        total_budgets = total_budgets + budgets*len(parallel_param)
        
        for m in range(len(results)):
            result_pop_acc.append(results[m]['loss'])
    
    return result_pop_acc



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
    f.write("Now time: %s\n" % time.strftime("%X",time.localtime()))



def add_trials(points):
    test_trials = Trials()
    
    for tid, row in enumerate(points):
        vals = {}
        for key in sample(space).keys():
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
        temp['status'] = STATUS_OK
        
        points.append(temp)
    
    return points
   


if __name__ == "__main__":
    total_start_time = time.strftime("%X",time.localtime())
    f.write("Start time: %s\n" % (total_start_time))
    budgets=first_budget
    population = []
    pop_from = []
    pop_size = init_pop_size
    
    print("###### RANDOM SAMPLING Training ######")
    start_time = time.time()
    # random initialize
    for i in range(pop_size):
        population.append(sample(space))
        pop_from.append("Random")
   
    pop_acc = parallel_training(population)

    end_time = time.time()
    f.write("============================================================")
    f.write("\nRandom Sampling %s sets with %s budgets Training time: %s sec\n" 
            %(pop_size, budgets, round(end_time-start_time, 4)))
    print_population(population, pop_acc, pop_from)
    
    for s in range(int(log(max_budgets/first_budget, etha))):
        print("\n###### BAYESIAN OPTIMIZATION Training %s phase, %s sets ######" 
              % (budgets, round(pop_size/etha)))
        # another 5 with BO
        tpe_algorithm = tpe.suggest
        
        bayes_trials = add_trials(convert_results_to_trials(population, pop_acc))
                
        start_time = time.time()
        best = fmin(fn = get_accuracy, space = space, algo = partial(tpe_algorithm, gamma=0.2, n_startup_jobs=0), 
                    max_evals = pop_size + round(pop_size/etha), trials = bayes_trials, 
                    rstate = np.random.RandomState(50))
        
        total_budgets = total_budgets + round(pop_size/etha)*budgets
        
        end_time = time.time()
        f.write("============================================================")
        f.write("\n###### BAYESIAN OPTIMIZATION sampling %s sets, trainied with %s budgets ######\n" 
              % (round(pop_size/etha), budgets))
      
        cc=1
        for i in bayes_trials.results[pop_size:]:
            population.append(i['params'])
            pop_acc.append(i['loss'])
            pop_from.append("Bayesian " + str(budgets))
            
            f.write("\nBayesian Optimization %sth set with %s budgets\n" % (cc, budgets))
            f.write("pop: %s\n" % i['params'])
            f.write("pop_from: Bayesian %s\n" % str(budgets))
            f.write("pop_acc: %s\n" % -i['loss'])
            f.write("Now time: %s\n" % time.strftime("%X",time.localtime()))
            cc+=1
        
        f.write("BO Training time: %s sec" %round(end_time-start_time, 4))
        f.write("\nTotal spend budgets: %s" % total_budgets)
        f.write("\n============================================================")
        
        print("###### Remove and retain %s sets in population ######" % round(pop_size/etha))
        sort = np.argsort(pop_acc)[:round(pop_size/etha)]
        
        new_population, new_pop_acc, new_pop_from = [], [], []
        
        for i in sort:
            new_population.append(population[i])
            new_pop_acc.append(pop_acc[i])
            new_pop_from.append(pop_from[i])
        
        population, pop_acc, pop_from = new_population, new_pop_acc, new_pop_from
        del(new_population, new_pop_acc, new_pop_from)
        
        pop_size = len(population)
        f.write("\nAfter insert BO, %s sets with %s budgets population results\n" % (pop_size, budgets))
        print_population(population, pop_acc, pop_from)
                   
        # Train epoch*etha
        budgets = budgets*etha
        if s == int(log(max_budgets/first_budget, etha))-1:
            budgets = max_budgets
        
        print("###### Population More Training to %s budgets ######" % budgets)
        start_time = time.time()
        
        new_pop_acc = parallel_training(population)
                
        pop_acc = new_pop_acc
        del(new_pop_acc)
        
        end_time = time.time()
        
        f.write("============================================================")
        #f.write("\nPopulation more training to %s epoch Training time: %s sec\n" 
        #        %(epochs, round(end_time-start_time, 4)))
        f.write("\nPopulation more training to %s budget" % budgets)
        f.write("\nTotal spend budgets: %s\n" % total_budgets)
        
        f.write("After More training to %s budgets in %s sets, population results\n" %(budgets, pop_size))
        print_population(population, pop_acc, pop_from)
        
    
    total_end_time = time.strftime("%X",time.localtime())
    f.write("\nEnd time: " + str(total_end_time))
    f.write("\nTotal spend budgets: %s\n" % total_budgets)
            
    f.close()
    