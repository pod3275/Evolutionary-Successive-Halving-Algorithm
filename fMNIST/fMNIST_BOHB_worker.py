# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:47:45 2019

@author: lawle
"""

try:
    from keras import backend as K
    from keras.datasets import fashion_mnist
    from sklearn import metrics
    from sklearn.svm import SVC
except:
    raise ImportError("For this example you need to install keras and scikit-learn.")


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import time
from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)


class SVMWorker(Worker):
    def __init__(self, *args, sleep_interval=0.5, **kwargs):
        super().__init__(*args, **kwargs)
    
        img_rows = 28
        img_cols = 28
        self.num_classes = 10
            
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
    
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.total_budgets = 0
        self.input_shape = (img_rows, img_cols, 1)

        self.sleep_interval = sleep_interval
        self.one_budget = 1000


    def compute(self, config, budget, *args, **kwargs):
        C = 10 ** config['C']
        gamma = 10 ** config['gamma']
        classifier = SVC(C=C, gamma=gamma, random_state=1234)

        # Use Keras to train the model.
        classifier.fit(self.x_train[:int(budget)*self.one_budget], self.y_train[:int(budget)*self.one_budget])
        accuracy = metrics.accuracy_score(self.y_test,
                                          classifier.predict(self.x_test))
            
        self.total_budgets = self.total_budgets + int(budget)
                    
        time.sleep(self.sleep_interval)
        
        return ({
        	'loss': -accuracy, # remember: HpBandSter always minimizes!
        	'info': {	
                        'time': time.strftime("%X",time.localtime()),
                        'configuration': config,
        			}
        						
        })


    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        C = CSH.UniformFloatHyperparameter('C', lower=-3, upper=2, default_value='0.7', log=False)
        gamma = CSH.UniformFloatHyperparameter('gamma', lower=-4, upper=-1, default_value='-2', log=False)
        
        cs.add_hyperparameters([C, gamma])

        return cs


if __name__ == "__main__":
    worker = SVMWorker(run_id='0')
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()

    worker.compute(config, budget=3)

    print(config)
    res = worker.compute(config=config, budget=1)

    print(res)
