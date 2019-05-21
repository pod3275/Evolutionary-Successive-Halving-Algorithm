# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:47:45 2019

@author: lawle
"""

try:
    import keras
    import tensorflow as tf
    from keras import backend as K
    from keras.datasets import cifar10
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    from keras.optimizers import Adam
except:
    raise ImportError("For this example you need to install keras.")


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import time
from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)


class KerasWorker(Worker):
    def __init__(self, *args, sleep_interval, **kwargs):
        super().__init__(*args, **kwargs)
    
        img_rows = 32
        img_cols = 32
        self.num_classes = 10
            
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = keras.utils.to_categorical(y_train, self.num_classes) 
            
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = keras.utils.to_categorical(y_test, self.num_classes)  
    
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.total_epochs = 0
        self.input_shape = (img_rows, img_cols, 3)

        self.sleep_interval = sleep_interval


    def compute(self, config, budget, *args, **kwargs):
         with tf.Session(graph=tf.Graph()) as sess:
            K.set_session(sess)
            
            model = Sequential()
            model.add(Conv2D(filters=config['num_filters_1'], kernel_size=(3, 3),
                             padding='same', activation='relu', input_shape=self.input_shape,
                             kernel_initializer='he_normal'))
            model.add(Conv2D(filters=config['num_filters_1'], kernel_size=(3, 3),
                             activation='relu', kernel_initializer='he_normal'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(config['dropout']))
            
            model.add(Conv2D(filters=config['num_filters_2'], kernel_size=(3, 3), 
                             activation='relu', padding='same', kernel_initializer='he_normal'))
            model.add(Conv2D(filters=config['num_filters_2'], kernel_size=(3, 3),
                             activation='relu', kernel_initializer='he_normal'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(config['dropout']))
            
            model.add(Flatten())
            for i in range(config['num_dense_layers']):
                model.add(Dense(config['num_dense_nodes'], activation=config['dense_activation'])) #1
                
            model.add(Dropout(config['dropout']))
            model.add(Dense(self.num_classes, activation='softmax'))
                   
            optimizer = Adam(lr=config['learning_rate'])
            
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            #------------------------------------------
            
            history = model.fit(x=self.x_train,
                                y=self.y_train,
                                epochs=int(budget),
                                batch_size=config['batch_size'],
                                validation_data=(self.x_test, self.y_test),
                                verbose=0)
            
            accuracy = history.history['val_acc'][-1]
            
            self.total_epochs = self.total_epochs + int(budget)
                        
            time.sleep(self.sleep_interval)
            
            return ({
            	'loss': -accuracy, # remember: HpBandSter always minimizes!
            	'info': {	
                            'configuration': config,
            				'number of parameters': model.count_params(),
            			}
            						
            })


    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-5, upper=1e-2, default_value='1e-3', log=False)
        batch_size = CSH.CategoricalHyperparameter('batch_size', [64,128,256])

        cs.add_hyperparameters([lr, batch_size])

        num_filters_1 = CSH.CategoricalHyperparameter('num_filters_1', [16,32,64,128])
        num_filters_2 = CSH.CategoricalHyperparameter('num_filters_2', [16,32,64,128])

        cs.add_hyperparameters([num_filters_1, num_filters_2])

        dropout_rate = CSH.UniformFloatHyperparameter('dropout', lower=0.2, upper=1.0, default_value=0.5, log=False)	
        num_dense_layers = CSH.UniformIntegerHyperparameter('num_dense_layers', lower=1, upper=4, default_value=1, log=False)
        num_dense_nodes = CSH.UniformIntegerHyperparameter('num_dense_nodes', lower=32, upper=512, default_value=128, log=False)
        dense_act = CSH.CategoricalHyperparameter('dense_activation', ['relu','sigmoid','tanh'])

        cs.add_hyperparameters([dropout_rate, num_dense_layers, num_dense_nodes, dense_act])

        return cs


if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()

    worker.compute(config, budget=6)

    print(config)
    res = worker.compute(config=config, budget=1)

    print(res)
