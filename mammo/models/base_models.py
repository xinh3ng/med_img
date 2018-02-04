# -*- coding: utf-8 -*-
"""Base model functionalities

"""
from pdb import set_trace as debug
from pydsutils.generic import create_logger


logger = create_logger(__name__, level='info')


class BaseModelOperator(object):
    """Base class of model operators

    """
    def __init__(self, input_shape, num_classes, include_top, weights, optimizer, loss, metrics):
        """
        
        Args:
            include_top: whether to include the 3 fully-connected layers at the top of the network.
            weights:
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
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
