# -*- coding: utf-8 -*-
"""
VGG16 model for Keras.

Reference:
    Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556
"""
from pdb import set_trace as debug
import sys
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
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
    

def gen_vgg16(include_top=True, weights='imagenet', input_tensor=None):
    """Instantiate the VGG16 architecture
    
    :param include_top: whether to include the 3 fully-connected layers at the top of the network.
    :param weights:
    :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                         to use as image input for the model.
    :return:             a Keras model instance.
    """
    logger.info('include_top is %s, weights is %s' % (str(include_top), weights))
            
    # Determine proper input shape based on tensorflow or theano as a backend
    assert K.image_dim_ordering() == 'tf', 'Bakend must be tensorflow but found otherwise' 
    if include_top:
        input_shape = (224, 224, 3)
        input_shape = (None, None, 3)

    if input_tensor is None:
        logger.info('input_tensor is None')
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    
    model = Sequential()
    # Block 1
    model.add(Conv2D(img_input, 16, 3, 3, border_mode='same', name='block1_conv1'))
    debug()
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(32, 3, 3, border_mode='same', name='block1_conv2'))
    model.add(PReLU())
    model.add(BatchNormalization())

    if include_top:
        # Classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(1, activation='sigmoid', name='predictions'))

    # load weights
    if weights is not None:
        pass

    return model
