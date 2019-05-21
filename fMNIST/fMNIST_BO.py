# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:45:44 2019

@author: lawle
"""

import numpy as np
import time
import os
from keras.datasets import fashion_mnist
from skopt import Optimizer
from skopt.space import Real
from tqdm import tqdm
from sklearn import metrics
from sklearn.svm import SVC

from keras import backend as K

n_observe = 30
one_budget = 1000
max_budget = 27

f = open('BO_fMNIST_%sobserve_%sbudget.txt' % (n_observe, max_budget), 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dim_C = Real(low=1e-3, high=1e2, prior='log-uniform',name='C')
dim_gamma = Real(low=1e-4, high=1e-1, prior='log-uniform',name='gamma')

dimensions = [dim_C, dim_gamma]


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
    classifier = SVC(C=params[0], gamma=params[1], random_state=1234)
    
    # Use Keras to train the model.
    print("Fitting SVM")
    start_time = time.time()
    classifier.fit(train_data[0][:max_budget*one_budget], train_data[1][:max_budget*one_budget])
    
    print("Get accuracy of SVM")
    accuracy = metrics.accuracy_score(validation_data[1], 
                                      classifier.predict(validation_data[0]))
    
    global total_budgets
    total_budgets = total_budgets + max_budget
    print("Result accuracy: %s" % accuracy)
    end_time = time.time()
    print("Consumed time: %ssec" % round(end_time-start_time, 4))
    
    del(classifier)
    return -accuracy


if __name__ == "__main__":
    total_start_time = time.strftime("%X",time.localtime())
    f.write("Start time: %s\n" % (total_start_time))
    population = []
    pop_acc = []
    
    global total_budgets
    total_budgets = 0
    
    print("\n###### BAYESIAN OPTIMIZATION Training %s observes ######" % (n_observe))
    f.write("###### BAYESIAN OPTIMIZATION Training %s observes ######\n" % (n_observe))
    # another 5 with BO
    
    opt = Optimizer(
            dimensions=dimensions,
            acq_func="EI",
            n_initial_points=5,
            random_state=1234
    )
        
    for i in tqdm(range(n_observe)):
        next_set = opt.ask()
        print("\nBayesian Optimization %sth" %(i+1))
        print("observer: %s\n" % next_set)
        start_time = time.time()
        next_acc = get_accuracy(next_set)
        end_time = time.time()
        f.write("\n%sth observation\n" % (i+1))
        f.write("observer: %s\n" % next_set)
        f.write("pop_acc: %s\n" % -next_acc)
        f.write("Time: %s, consuming: %s seconds\n" 
                % (time.strftime("%X",time.localtime()), round(end_time-start_time, 4)))
        
        opt.tell(next_set, next_acc)
        population.append(next_set)
        pop_acc.append(next_acc)
        
    f.write("============================================================")
    f.write("\nBest accuracy: %s\n" %-min(pop_acc))
    f.write("Best param set: %s\n" % (population[np.argmin(pop_acc)]))
         
    total_end_time = time.strftime("%X",time.localtime())
    f.write("\nEnd time: " + str(total_end_time))
    f.write("\nTotal budgets: %s" % total_budgets)
    f.close()