# -*- coding: utf-8 -*-
from pdb import set_trace as debug
from typing import Any, Tuple
import numpy as np
import keras
from keras.datasets import mnist, cifar10
from pydsutils.generic import create_logger

logger = create_logger(__name__)


def load_image_data(dataset):
    """Load the images as numpy arrays, reshape them accordingly
    
    X's shape should be (num_samples, height, width, channel)
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), i.e. numpy arrays or None
    """
    fn = {'mnist': mnist.load_data,
          'cifar10': cifar10.load_data
          }[dataset]
  
    def func(input_shape: tuple, *args: Any, **kwargs: Any) -> \
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array], Any]:
        (X_train, y_train), (X_val, y_val) = fn()
    
        # Reshape X to (samples, rows, columns)
        X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1], input_shape[2]).astype('float32')
        X_val = X_val.reshape(X_val.shape[0], input_shape[0], input_shape[1], input_shape[2]).astype('float32')
        X_train /= 255.0  # data preprocessing
        X_val /= 255.0
        
        # Hot encode it into 10 columns
        y_train = keras.utils.to_categorical(y_train, 10)
        y_val = keras.utils.to_categorical(y_val, 10)
        return (X_train, y_train), (X_val, y_val), (None, None)

    return func

