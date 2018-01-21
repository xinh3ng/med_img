# -*- coding: utf-8 -*-
"""
VGG16 model for Keras.

Reference:
    Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556
"""
from pdb import set_trace as debug
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.layers import Flatten, Dense, Input
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential

from med_img.mammo.utils.generic_utils import create_logger
# from mammo_vgg_relu import gen_vgg16
 
logger = create_logger(__name__, level='info')


def get_vgg16_model(use_relu):
    """Return the chosen VGG16 model"""
    if use_relu:
        raise NotImplementedError
    else:
        return gen_vgg16
    

def gen_vgg16(input_shape, weights='imagenet',
              optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
    """Instantiate the VGG16 architecture
    
    :param include_top: whether to include the 3 fully-connected layers at the top of the network.
    :param weights:
    """
    logger.info('input_shape is %s, weights is %s' % (str(input_shape), weights))
            
    # Determine proper input shape based on tensorflow or theano as a backend
    assert K.image_dim_ordering() == 'tf', 'Backend must be tensorflow but found otherwise'
    
    model = Sequential()
    
    # Block 1
    model = Sequential()

    # 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))


    # load weights
    if weights is not None:
        pass
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=metrics)
    return model
