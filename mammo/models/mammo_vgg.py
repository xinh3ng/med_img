# -*- coding: utf-8 -*-
"""
VGG16 model for Keras.

Reference:
    Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556
"""
from pdb import set_trace as debug
from keras import backend as K
from keras.applications.vgg16 import VGG16
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
            
    # Determine proper input shape based on tensorflow or theano as a backend
    assert K.backend() == 'tensorflow', 'Backend must be tensorflow but found otherwise'
    
    model = VGG16(include_top=True, weights=weights,
                  input_shape=input_shape, pooling=None,
                  classes=classes)
    
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=metrics)
    logger.info("Successfully compiled the model")
    return model
