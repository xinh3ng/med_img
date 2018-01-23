# -*- coding: utf-8 -*-
"""

Models include:
    CNN
    VGG16 model
"""
from pdb import set_trace as debug
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input

from med_img.mammo.utils.generic_utils import create_logger

 
logger = create_logger(__name__, level='info')
assert K.backend() == 'tensorflow', 'Backend must be tensorflow but found otherwise'


def select_model_processor(model_name):
    """Factory function that selects the model"""
    x = {'vgg16': VGG16ModelProcessor,
         'cnn': SimpleCNNModelProcessor
         }
    return x.get(model_name, BaseModelProcessor)


class BaseModelProcessor(object):
    def __init__(self, input_shape, classes, include_top, weights, optimizer, loss, metrics):
        """
        
        Args:
            include_top: whether to include the 3 fully-connected layers at the top of the network.
            weights:
        """
        self.input_shape = input_shape
        self.classes = classes
        self.include_top = include_top
        self.weights = weights
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        logger.info('input_shape is %s, weights is %s' % (str(self.input_shape), self.weights))
        logger.info('optimizer, loss, and metrics: %s, %s, %s' % (str(self.optimizer), 
                str(self.loss), str(self.metrics)))
    
    def create_model(self):
        raise NotImplementedError
    
    def process_X(self, X):
        return X
    
    def process_y(self, y):
        return y


class VGG16ModelProcessor(BaseModelProcessor):
    def __init__(self, input_shape, classes,
                 include_top=True, weights=None,
                 optimizer='adam', loss='categorical_crossentropy', 
                 metrics=['accuracy']):
        super(VGG16ModelProcessor, self).__init__(input_shape, classes, include_top, weights,
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
        return preprocess_input(X)
    

class SimpleCNNModelProcessor(BaseModelProcessor):
    def __init__(self, input_shape, classes,
                 include_top=True, weights=None,
                 optimizer='adam', loss='categorical_crossentropy', 
                 metrics=['accuracy']):
        super(SimpleCNNModelProcessor, self).__init__(input_shape, classes, include_top, weights,
                 optimizer, loss, metrics)

    def create_model(self, verbose=0):
        """Instantiate a simple CNN model
        
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

