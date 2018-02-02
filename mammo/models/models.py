# -*- coding: utf-8 -*-
"""

Models include:
  CNN
  VGG16
Usage:
  $ python train_mammo_model.py --dataset_name=mias --model_name=tfvgg16 --optimizer=adam --loss=categorical_crossentropy
"""
from pdb import set_trace as debug
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input
import torchvision.models

from pydsutils.generic import create_logger

from med_img.mammo.models.base_models import BaseModelOperator


logger = create_logger(__name__, level='info')

#####################################################
# Model operators. Goals include:
#   Preprocess input data: normalize, one-hot encoding, outliers, etc.
#   Create a model of choice
#####################################################

class TfVgg16ModelOperator(BaseModelOperator):
    def __init__(self, input_shape, classes,
                 include_top=True, weights=None,
                 optimizer='adam', loss='categorical_crossentropy', 
                 metrics=['accuracy']):
        super(TfVgg16ModelOperator, self).__init__(input_shape, classes, include_top, weights,
                                                 optimizer, loss, metrics)
        
    def create_model(self, verbose=0):
        """Instantiate the VGG16 architecture
    
        """
        model = VGG16(include_top=self.include_top, weights=self.weights,
                      input_shape=self.input_shape, pooling=None,
                      classes=self.classes)
        
        model.compile(optimizer=self.optimizer, loss=self.loss,
                      metrics=self.metrics)
        logger.info("Successfully compiled the model")
        if verbose >= 1:
            logger.info("Print model summary:")
            model.summary()  # print model summary
        return model
    
    def process_X(self, X):
        return preprocess_input(X)  # for vgg16 only
    

class TfSimpleCnnModelOperator(BaseModelOperator):
    def __init__(self, input_shape, classes,
                 include_top=True, weights=None,
                 optimizer='adam', loss='categorical_crossentropy', 
                 metrics=['accuracy']):
        super(TfSimpleCnnModelOperator, self).__init__(input_shape, classes, include_top, weights,
                                                     optimizer, loss, metrics)

    def create_model(self, verbose=0):
        """Instantiate a simple Cnn model
        
        https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
        """    
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes, activation='softmax'))
    
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        logger.info("Successfully compiled the model")
        if verbose >= 1:
            logger.info("Print model summary:")
            model.summary()  # print model summary
        return model


class TorchVgg16ModelOperator(BaseModelOperator):
    def __init__(self, input_shape, classes,
                 include_top=True, weights=None,
                 optimizer='adam', loss='categorical_crossentropy',
                 metrics=['accuracy']):
        super(TorchVgg16ModelOperator, self).__init__(input_shape, classes, include_top, weights,
                                                       optimizer, loss, metrics)

    def create_model(self, verbose=0):

        model = torchvision.models.vgg16()
        if verbose >= 1:
            logger.info("Print model summary:")
            model.summary()  # print model summary
        return model

#####################################################

class ModelOutput(object):
    """Class to standardize the model output

    """
    def __init__(self, model_output):
        self.model_output = model_output  # model_output in its native tf or torch format
        self.history = model_output.get("history", None)

    @property
    def history(self):
        return self.history

