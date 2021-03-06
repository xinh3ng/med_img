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
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pydsutils.generic import create_logger
from pydsutils.file_utils import ensure_dir_exists
from pymlearn.dl_utils import TfMemoryUsage, TfMetrics

from med_img.mammo.config import data_configs as dc
from med_img.mammo.config import model_configs as mc
from med_img.mammo.utils.data_utils import gen_img_sets_table, batch_gen_model_data
from med_img.mammo.models.model_utils import select_model, select_model_data_validator

logger = create_logger(__name__, level='info')


def gen_callbacks(filename, metric, min_delta, verbose=1):
    """Generate callback functions
    """
    memory = TfMemoryUsage(show_batch_begin=False, show_batch_end=False)
    # metrics = TfMetrics()
    checkpt = ModelCheckpoint(filename, monitor=metric, save_best_only=False, period=1, verbose=verbose)
    stopping = EarlyStopping(monitor=metric, min_delta=min_delta, patience=1, verbose=verbose)
    # return [memory, metrics, checkpt, stopping]
    return [memory]


def save_train_metrics(history, filename, metrics = ['loss', 'acc']):
    """Save metrics from the training process for later visualization"""
    result = pd.DataFrame()
    for metric in metrics:
        result[metric] = history.history[metric]
        result['val_' + metric] = history.history['val_' + metric]

    result.to_csv(filename, index=False)
    logger.info("Successfully saved metrics from the training process in: " + filename)


def main(data_src='mnist', sample_sizes='0,0,0', model_name='tfcnn',
         val_pct=0.1, test_pct=0.0,  n_jobs=-1,
         epochs=5,
         optimizer='adam', loss='categorical_crossentropy'):
    """Main function
    
    Args:
        data_src: Data source: mnist or mias or ddsm. Config file is indexed by using this field')
        sample_sizes: Total size of training and validation sets. 0 means all samples
        model_name: Clf model: cnn or vgg16
        val_pct: Pct of data as validation set: 0.1
        optimizer: Optimizer algorithm: adam
        loss: Loss function used by optimizer: sparse_categorical_crossentropy, categorical_crossentropy
              or binary_crossentropy
    """
    sample_sizes = tuple([int(x) for x in sample_sizes.split(',')])
    input_shape = dc.src_data_configs[data_src]['input_shape']
    batch_size = mc.model_configs[data_src]['batch_size']

    # Create train / val / test data generators
    img_sets_table = gen_img_sets_table(data_src, val_pct, test_pct, verbose=1)
    md_iters = {}  # model data iterators
    for type in ['train', 'val', 'test']:
        md_iters[type] = batch_gen_model_data(
            data_src, img_sets_table, type, sample_sizes, input_shape,
            n_jobs=n_jobs, batch_size=batch_size)

    # Generate the model instance
    num_classes = len(dc.src_data_configs[data_src]['labels'].keys())  # number of classes
    tf_model = select_model(model_name)(
        input_shape=input_shape, num_classes=num_classes, batch_size=batch_size,
        include_top=True, weights=None,
        optimizer=optimizer, loss=loss)
    train_length = sum(img_sets_table['type'] == 'train')  # Get train and validation data length
    val_length = sum(img_sets_table['type'] == 'val')

    ensure_dir_exists(dc.model_state_dir.format(data_src=data_src))
    callbacks = gen_callbacks(dc.model_state_dir.format(data_src=data_src) +\
            '/{epoch:04d}_{val_loss:.4f}_' + dc.model_filename, 
            metric='val_loss',
            min_delta=0.001, verbose=2)

    logger.info('Start the fitting process...')
    history = tf_model.fit(
        data=md_iters['train'],
        train_length=train_length,
        epochs=epochs,
        val_data=md_iters['val'],
        val_length=val_length,
        callbacks=callbacks)
    logger.info("Successfully fit the model")

    tf_model.evaluate(md_iters['val'], val_length=val_length)
    # tf_model.save(dc.model_state_dir.format(data_src=data_src) + '/' + dc.model_filename)
    logger.info('Successfully saved the model')

    # Save metrics from the training process for later visualization
    save_train_metrics(history,
        filename=dc.model_state_dir.format(data_src=data_src) + '/train-' + data_src + '.csv')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_src', default='mnist', 
            help='Data source: mnist, cifar10, or mias or ddsm')
    parser.add_argument('--sample_sizes', default='0,0,0',
            help='Total size of train, validation and test')
    parser.add_argument('--model_name', default='tfcnn',
            help='Clf model: tfcnn or tfvgg16 or torchcnn')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--val_pct', type=float, default=0.1, help='Pct of data as validation set')
    parser.add_argument('--optimizer', default='adam',
            help='Optimizer algorithm: adam')
    parser.add_argument('--loss', default='categorical_crossentropy',
            help='Loss function used by optimizer')
    args = parser.parse_args()
    main(**vars(args))
    logger.info('ALL DONE\n')
