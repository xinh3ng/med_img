# -*- coding: utf-8 -*-
"""

"""
from pdb import set_trace as debug
import keras.backend as K
from pydsutils.generic import create_logger

logger = create_logger(__name__, 'info')


def precision(y_true, y_pred):
    """Precision. Only computes a batch-wise average of precision, which can be misleading

    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall. Only computes a batchwise average of recall.

    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))  # TP + FN
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

