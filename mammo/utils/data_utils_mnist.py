# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import keras
from keras.datasets import mnist
from med_img.mammo.utils.generic_utils import create_logger

logger = create_logger(__name__)


def load_image_data(image_dir, labels, val_pct, test_pct,
                    input_shape):
    """Load the images as numpy arrays, reshape them accordingly
    
    X's shape should be (num_samples, height, width, channel)
    """
    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    
    # Reshape X to (samples, rows, columns, 1)
    X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
    X_val = X_val.reshape(X_val.shape[0], input_shape[0], input_shape[1], input_shape[2])
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_val /= 255
    
    # 
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    
    logger.info("Successfully loaded image files as numpy arrays. Shape of X_train and y_train are: %s, %s"\
                % (str(X_train.shape), str(y_train.shape)))
    logger.info("Shape of X_val and y_val are: %s, %s" % (str(X_val.shape), str(y_val.shape)))
    
    return (X_train, y_train), (X_val, y_val), (None, None)

