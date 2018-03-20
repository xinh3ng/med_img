# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import pandas as pd
import keras
from keras.datasets import mnist, cifar10
from pydsutils.generic import create_logger

logger = create_logger(__name__)


def create_img_sets(*args, **kwargs):
    """An empty function. Needed to keep consistent interfaces

    """
    return pd.DataFrame(columns=['filename' , 'label', 'label_num', 'type'])


def gen_model_data(data_src):
    """Load the images as numpy arrays, reshape them accordingly
    
    X's shape should be (num_samples, height, width, channel)
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), i.e. numpy arrays or None
    """
    fn = {'mnist': mnist.load_data,
          'cifar10': cifar10.load_data
          }[data_src]
  
    def func(type, sample_size, input_shape, batch_size, **kwargs):
        assert batch_size == 0, 'Load all data at same time, so batch_size must set to 0'
        if type == 'test':
            return None, None

        (X_train, y_train), (X_val, y_val) = fn()
        if type == 'train':
            X, y = X_train, y_train
        elif type == 'val':
            X, y = X_val, y_val
    
        # Reshape X to (samples, rows, columns), sp there must be sample size requirement
        assert sample_size == 0 or sample_size <= X.shape[0], 'Requested data size is too big'
        
        size = X.shape[0] if sample_size == 0 else sample_size
        X = X.reshape(X.shape[0], input_shape[0], input_shape[1], input_shape[2]).astype('float32')
        X = X[0:size,::]
        X /= 255.0  # data pre-processing
        
        # Hot encode it into 10 columns
        y = keras.utils.to_categorical(y[0:size], 10)
        logger.info('Shape of X for %s is: %s' % (type, str(X.shape)))
        logger.info('Shape of y for %s is: %s' % (type, str(y.shape)))
        return X, y

    return func

