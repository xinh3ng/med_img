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

from med_img.mammo.utils.generic_utils import create_logger

 
logger = create_logger(__name__, level='info')
assert K.backend() == 'tensorflow', 'Backend must be tensorflow but found otherwise'


def gen_model(model_name):
    """Factory function that selects the model"""
    x = {'vgg16': gen_vgg16,
         'cnn': gen_cnn
         }
    return x[model_name]


def gen_vgg16(input_shape, classes,
              include_top=True, weights=None,
              optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy']):
    """Instantiate the VGG16 architecture
    
    Args:
        include_top: whether to include the 3 fully-connected layers at the top of the network.
        weights:
    """
    logger.info('input_shape is %s, weights is %s' % (str(input_shape), weights))
    logger.info('optimizer, loss, and metrics: %s, %s, %s' % (str(optimizer), 
            str(loss), str(metrics)))
    
    model = VGG16(include_top=include_top, weights=weights,
                  input_shape=input_shape, pooling=None,
                  classes=classes)
    
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=metrics)
    logger.info("Successfully compiled the model")
    return model


def gen_cnn(input_shape, classes, weights=None,
            optimizer='adam', loss='categorical_crossentropy', 
            metrics=['accuracy'], **kwargs):
    """Instantiate a simple CNN model
    
    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    Args:
        weights:
    """
    logger.info('input_shape is %s, weights is %s' % (str(input_shape), weights))
    logger.info('optimizer, loss, and metrics: %s, %s, %s' % (str(optimizer), 
            str(loss), str(metrics)))
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    logger.info("Successfully compiled the model")
    return model
