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
import time
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

from sklearn import metrics
from sklearn.svm import SVC

from keras import backend as K
from keras.datasets import fashion_mnist

n_iterations = 30
one_budget = 1000
max_budget = 27

f = open('TPE_fMNIST_%sobserve_%sbudgets.txt' % (n_iterations, max_budget), 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the search space
space = {
    'C': hp.uniform('C', -3, 2),
    'gamma': hp.uniform('gamma', -4, -1)
}


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
    global ITERATION
    ITERATION += 1
    
    C = 10 ** params['C']
    gamma = 10 ** params['gamma']
    classifier = SVC(C=C, gamma=gamma, random_state=1234)
    
    start = timer()
    
    print("Fitting SVM")
    # Use Keras to train the model.
    classifier.fit(train_data[0][:max_budget*one_budget], train_data[1][:max_budget*one_budget])
    
    print("Get accuracy of SVM")
    accuracy = metrics.accuracy_score(validation_data[1],
                                      classifier.predict(validation_data[0]))
    run_time = timer() - start
    now_time = time.strftime("%X",time.localtime())
    global TOTAL_EPOCHS
    TOTAL_EPOCHS = TOTAL_EPOCHS + max_budget
    
    loss = -accuracy
    print("Accuracy: %s" % accuracy)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([accuracy, {'C': C, 'gamma': gamma}, ITERATION, now_time, run_time])
    
    return {'loss': loss, 'params': params, 'status': STATUS_OK}



if __name__ == "__main__":
    total_start_time = time.strftime("%X",time.localtime())
    f.write("Start time: %s\n" % (total_start_time))
    # optimization algorithm
    tpe_algorithm = tpe.suggest
    
    # Keep track of results
    bayes_trials = Trials()
    
    # File to save first results
    out_file = ('TPE_fMNIST_%sobserve.csv' % n_iterations)
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)
    
    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration', 'now_time', 'train_time'])
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

    results = pd.read_csv('TPE_fMNIST_%sobserve.csv' % n_iterations)
    
    # Sort with best scores on top and reset index for slicing
    results.sort_values('loss', ascending = False, inplace = True)
    results.reset_index(inplace = True, drop = True)

    f.write("Best parameter set : %s" % results.loc[0, 'params'])
    f.write("\n\nBest accuracy : %s" % results.loc[0, 'loss'])
    f.write("\n\nBest set is %sth iteration" % results.loc[0, 'iteration'])
    f.write("\n\n\nTotal epochs : %s" % TOTAL_EPOCHS)
    
    total_end_time = time.strftime("%X",time.localtime())
    f.write("\n\nEnd time: " + str(total_end_time))
    
    f.close()
