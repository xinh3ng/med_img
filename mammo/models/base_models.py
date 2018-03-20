# -*- coding: utf-8 -*-
"""Base model functions

"""
from pdb import set_trace as debug
import types
import json
import numpy as np
from pydsutils.generic import create_logger

logger = create_logger(__name__, level='info')


class BaseTfModel(object):
    """Base class of model

    """
    def __init__(self, input_shape, num_classes, batch_size, include_top, weights,
                 optimizer, loss):
        """
        
        Args:
            include_top: whether to include the 3 fully-connected layers at the top of the network.
            weights:
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.include_top = include_top
        self.weights = weights
        self.optimizer = optimizer
        self.loss = loss
        logger.info('input_shape is %s, weights is %s' % (str(self.input_shape), self.weights))
        logger.info('batch_size, %d' % (self.batch_size))
        logger.info('optimizer, loss: %s, %s' % (str(self.optimizer), str(self.loss)))


        self.model = self.create_model()
        logger.info("Print model summary:")
        self.model.summary()

    
    def create_model(self):
        raise NotImplementedError
    
    def process_X(self, X, verbose=0):
        return X
    
    def process_y(self, y, verbose=0):
        return y

    def fit(self, data, train_length, epochs,
            val_data, val_length,
            callbacks):
        """Fit the model using either fit() or fit_generator()
        """
        if self.batch_size == 0:  # Load in the whole data
            # Use the default batch_size, which is 32
            history = self.model.fit(x=data[0], y=data[1], validation_data=val_data,
                                     epochs=epochs, callbacks=callbacks,
                                     verbose=1)
        else:  # Use data generator
            assert isinstance(data, types.GeneratorType)
            history = self._fit_generator(data, train_length, epochs,
                                          val_data, val_length, callbacks, verbose=1)
        return history

    def _fit_generator(self, data, train_length, epochs,
                       val_data, val_length, callbacks, verbose):
        steps_per_epoch = train_length // self.batch_size
        validation_steps = val_length // self.batch_size
        assert validation_steps >= 1, 'must be larger than 1'

        history = self.model.fit_generator(data,
                                           steps_per_epoch=steps_per_epoch, epochs=epochs,
                                           validation_data=val_data, validation_steps=validation_steps,
                                           callbacks=callbacks,
                                           verbose=verbose)
        return history


    def evaluate(self, val_data, val_length):
        # Measure model performance with validation set

        if self.batch_size == 0:
            scores = self.model.evaluate(val_data[0], val_data[1], verbose=1)

        else:
            validation_steps = val_length // self.batch_size
            scores = self.model.evaluate_generator(
                val_data, steps=validation_steps)
        scores = {k: np.round(v, 4) for k, v in zip(self.model.metrics_names, scores)}
        logger.info('Perf scores on val data:\n%s' % json.dumps(scores, sort_keys=True, indent=4))
        return
