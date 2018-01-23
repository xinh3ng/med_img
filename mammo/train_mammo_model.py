# -*- coding: utf-8 -*-
"""

Procedure:
    Invoke virtual env (Python 3.6)
    $ python train_mammo_model.py --dataset_name=mnist --model_name=cnn --optimizer=adam --loss=categorical_crossentropy
"""
from pdb import set_trace as debug
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint

from med_img.mammo.utils.generic_utils import create_logger
import med_img.mammo.config.constants as c
from med_img.mammo.config.config_data import data_config
import med_img.mammo.utils.data_utils_ddsm as ddsm
import med_img.mammo.utils.data_utils_mias as mias
import med_img.mammo.utils.data_utils_mnist as mnist
from med_img.mammo.models.models import select_model_processor

logger = create_logger(__name__, level='info')
seed = os.getenv('RANDOM_SEED', 1337)  # Must use the RANDOM_SEED env as specified in challenge guidelines.


def load_image_data_fn(dataset_name):
    """Select the function that will load the image file into numpy arrays"""
    x = {'ddsm': ddsm.load_image_data,
         'mias': mias.load_image_data,
         'mnist': mnist.load_image_data
         }
    return x[dataset_name]


def gen_model_checkpoint():
    """Create model checkpoint callback for model training"""
    return ModelCheckpoint(
            c.MODELSTATE_DIR + '/{epoch:02d}' + c.MODEL_FILENAME,
            monitor='val_loss', save_best_only=False, save_weights_only=False,
            mode='auto', period=1, verbose=1)


def gen_early_stopping():
    return EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, 
                         verbose=1, mode='auto')

    
def save_train_metrics(history, metrics, filename):
    """Save metrics from the training process for later visualization"""

    with open(filename, 'w') as f:
        for metric in metrics:
            f.write('train {}\n'.format(metric))
            f.write(', '.join(str(x) for x in history.history[metric]))
            f.write('\n')
            f.write('validation {}\n'.format(metric))
            f.write(', '.join(str(x) for x in history.history['val_' + metric]))
            f.write('\n\n\n')
    logger.info("Successfully saved metrics from the training process in " + filename)


def main(dataset_name='mias', model_name='vgg16',
         use_relu=False, val_pct=0.1, batch_size=32, epochs=25,
         optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    """Main function
    
    Args:
        dataset_name: Data source: mnist or mias or ddsm. Config file is indexed by using this field')
        model_name: Clf model: cnn or vgg16
        val_pct: Pct of data as validation set: 0.1
        optimizer: Optimizer algorithm: adam
        loss: Loss function used by optimizer: categorical_crossentropy or binary_crossentropy
    """    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_image_data_fn(dataset_name)(
            image_dir=data_config[dataset_name]['data_dir'], 
            labels=data_config[dataset_name]['labels'],
            val_pct=val_pct,
            test_pct=0.0,
            input_shape=data_config[dataset_name]['input_shape'])
    
    # X = preprocess_input(X)  # this is for VGG model
    
    # Generate the model instance
    classes = len(data_config[dataset_name]['labels'].keys())  # number of classes
    model_processor = select_model_processor(model_name)(
            input_shape=data_config[dataset_name]['input_shape'], classes=classes,
            weights=None, optimizer=optimizer, loss=loss, metrics=metrics)
    model = model_processor.create_model(verbose=1)
    
    checkpt = gen_model_checkpoint()
    early_stopping = gen_early_stopping()
    
    X_train, y_train = model_processor.process_X(X_train), model_processor.process_y(y_train)
    X_val, y_val = model_processor.process_X(X_val), model_processor.process_y(y_val)    
    history = model.fit(x=X_train, y=y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping, checkpt])
    logger.info("Successfully fit the model")
    
    score = model.evaluate(X_val, y_val, verbose=0)
    logger.info('Validation loss: {}'.format(score[0]))
    logger.info('Validation accuracy: {}'.format(score[1]))
    
    model.save(c.MODELSTATE_DIR + '/' + c.MODEL_FILENAME)
    logger.info("Successfully saved the model")

    # Save metrics from the training process for later visualization
    save_train_metrics(history, metrics, filename=c.MODELSTATE_DIR + '/plot.txt')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='mias',
                        help='Data source: mnist or mias or ddsm')
    parser.add_argument('--model_name', default='cnn',
                        help='Clf model: cnn or vgg16')
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help='Pct of data as validation set')
    parser.add_argument('--optimizer', default='adam',
                        help='Optimizer algorithm: adam')
    parser.add_argument('--loss', default='categorical_crossentropy',
            help='Loss function used by optimizer: categorical_crossentropy or binary_crossentropy')

    args = parser.parse_args()
    main(**vars(args))
    logger.info('ALL DOE\n')
