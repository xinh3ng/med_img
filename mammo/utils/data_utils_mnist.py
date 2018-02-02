# -*- coding: utf-8 -*-
from pdb import set_trace as debug
from typing import Any, Tuple
import numpy as np
import keras
from keras.datasets import mnist
from pydsutils.generic import create_logger

logger = create_logger(__name__)


def load_image_data(input_shape: tuple, *args: Any, **kwargs: Any) -> \
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array], Any]:
    """Load the images as numpy arrays, reshape them accordingly
    
    X's shape should be (num_samples, height, width, channel)
    Returns:
        (X_train, y_train), (X_val, y_val), (None, None), i.e. numpy arrays or None
    """
    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    
    # Reshape X to (samples, rows, columns, 1)
    X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1], input_shape[2]).astype('float32')
    X_val = X_val.reshape(X_val.shape[0], input_shape[0], input_shape[1], input_shape[2]).astype('float32')
    X_train /= 255.0  # data preprocessing
    X_val /= 255.0
    
    # Hot encode it into 10 columns
    y_train = keras.utils.to_categorical(y_train, 10)  # mnist data has 10 classes in total
    y_val = keras.utils.to_categorical(y_val, 10)
    
    logger.info("Successfully loaded image files as numpy arrays. Shape of X_train and y_train are: %s, %s"\
                % (str(X_train.shape), str(y_train.shape)))
    logger.info("Shape of X_val and y_val are: %s, %s" % (str(X_val.shape), str(y_val.shape)))
    
    return (X_train, y_train), (X_val, y_val), (None, None)
