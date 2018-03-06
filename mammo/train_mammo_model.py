#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the mammograms

PROCEDURE:
  $ source venv/bin/activate (python 3.6)
  $ source scripts/setenv.sh (Set environement variables like PYTHONPATH
  $ cd mammo/
  $ python train_mammo_model.py --data_src=mnist --model_name=tfcnn --optimizer=adam --loss=categorical_crossentropy
"""
from pdb import set_trace as debug
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pydsutils.generic import create_logger
from pymlearn.dl_utils import TfMemoryUsage

from med_img.mammo.config import data_config as dc
from med_img.mammo.utils.data_utils import load_image_data_fn
from med_img.mammo.models.model_operation import select_model_operator, select_model_data_validator

logger = create_logger(__name__, level='info')
np.set_printoptions(precision=4)


def load_data(data_src, sample_sizes, val_pct, test_pct, input_shape):
    """Load the model data from factory function: load_image_data_fn
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_image_data_fn(data_src)(
            sample_sizes=sample_sizes,
            val_pct=val_pct, test_pct=0.0,
            input_shape=input_shape)
    logger.info("Successfully loaded image files as numpy arrays. Shape of X_train and y_train are: %s, %s"\
                % (str(X_train.shape), str(y_train.shape)))
    logger.info("Shape of X_val and y_val are: %s, %s" % (str(X_val.shape), str(y_val.shape)))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def gen_callbacks(filename, metric, min_delta, verbose=1):
    """Generate callback functions
    """
    checkpt = ModelCheckpoint(filename, monitor=metric, save_best_only=False, period=1,
            verbose=verbose)
    stopping = EarlyStopping(monitor=metric, min_delta=min_delta, patience=1, 
            verbose=verbose)
    memory = TfMemoryUsage(show_batch_begin=False, show_batch_end=False)
    return [memory, checkpt, stopping]

    
def save_train_metrics(history, metrics, filename):
    """Save metrics from the training process for later visualization"""

    with open(filename, 'w') as f:
        metrics = ['loss']
        for metric in metrics:
            f.write('train {}\n'.format(metric))
            f.write(', '.join(str(x) for x in history.history[metric]))
            f.write('\n')
            f.write('validation {}\n'.format(metric))
            f.write(', '.join(str(x) for x in history.history['val_' + metric]))
            f.write('\n')
    logger.info("Successfully saved metrics from the training process in file: " + filename)


def main(data_src='mnist', sample_sizes='0,0', model_name='vgg16',
         use_relu=False, val_pct=0.1, batch_size=32, epochs=25,
         optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    """Main function
    
    Args:
        data_src: Data source: mnist or mias or ddsm. Config file is indexed by using this field')
        sample_sizes: Total size of training and validation sets. 0 means all samples
        model_name: Clf model: cnn or vgg16
        val_pct: Pct of data as validation set: 0.1
        optimizer: Optimizer algorithm: adam
        loss: Loss function used by optimizer: categorical_crossentropy or binary_crossentropy
    """
    sample_sizes = tuple([int(x) for x in sample_sizes.split(',')])
    input_shape = dc.src_data_configs[data_src]['input_shape']
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
            data_src, sample_sizes,
            val_pct=val_pct, test_pct=0.0,
            input_shape=input_shape)
    
    # Generate the model instance
    num_classes = len(dc.src_data_configs[data_src]['labels'].keys())  # number of classes
    model_operator = select_model_operator(model_name)(
            input_shape=input_shape, num_classes=num_classes,
            weights=None, optimizer=optimizer, loss=loss, metrics=metrics)
    model = model_operator.create_model(verbose=1)
    
    callbacks = gen_callbacks(dc.model_state_dir.format(data_src=data_src) +\
            '/{epoch:04d}_{val_loss:.4f}_' + dc.model_filename, 
            metric='val_loss', 
            min_delta=0.001, verbose=2)
    X_train, y_train = model_operator.process_X(X_train), model_operator.process_y(y_train)
    X_val, y_val = model_operator.process_X(X_val), model_operator.process_y(y_val)
    
    model_data_validator = select_model_data_validator(model_name, num_classes=num_classes,
        num_rows=input_shape[0], num_columns=input_shape[1], num_channels=input_shape[2])
    model_data_validator.validate_X(X_train)
    model_data_validator.validate_y(y_train)
    model_data_validator.validate_X(X_val)
    model_data_validator.validate_y(y_val)
    
    logger.info("Data sets verified, now start the fitting process...")
    history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                        verbose=2)
    logger.info("Successfully fit the model")
    
    score = model.evaluate(X_val, y_val, verbose=0)
    logger.info('Validation loss: {0:.6f}'.format(score[0]))
    logger.info('Validation accuracy: {0:.6f}'.format(score[1]))
    
    model.save(dc.model_state_dir.format(data_src=data_src) + '/' + dc.model_filename)
    logger.info("Successfully saved the model")

    # Save metrics from the training process for later visualization
    save_train_metrics(history, metrics, filename=dc.model_state_dir.format(data_src=data_src)\
            + '/plot.txt')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_src', default='mnist', 
            help='Data source: mnist cifar10, or mias or ddsm')
    parser.add_argument('--sample_sizes', default='0,0', 
            help='Total size of train and validation')
    parser.add_argument('--model_name', default='tfcnn',
            help='Clf model: tfcnn or tfvgg16 or torchcnn')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_pct', type=float, default=0.1, help='Pct of data as validation set')
    parser.add_argument('--optimizer', default='adam',
            help='Optimizer algorithm: adam')
    parser.add_argument('--loss', default='categorical_crossentropy',
            help='Loss function used by optimizer: categorical_crossentropy or binary_crossentropy')

    args = parser.parse_args()
    main(**vars(args))
    logger.info('ALL DONE\n')
