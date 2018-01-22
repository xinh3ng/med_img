# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

from med_img.mammo.utils.generic_utils import create_logger
import med_img.mammo.config.constants as c
from med_img.mammo.config.config_data import data_config
# import med_img.mammo.utils.simple_loader as sl
from med_img.mammo.utils.data_utils import create_image_sets, load_image_data
from med_img.mammo.models.mammo_vgg import get_vgg16_model

logger = create_logger(__name__, level='info')
seed = os.getenv('RANDOM_SEED', 1337) # Must use the RANDOM_SEED environment as specified in challenge guidelines.


def gen_model_checkpoint():
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


def main(dataset_name='ddsm',
         use_relu=False, val_pct=0.1, batch_size=32, epochs=25,
         optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    """Main function

    """
    np.random.seed(int(seed))

    classes = len(data_config[dataset_name]['labels'].keys())
    
    image_sets = create_image_sets(image_dir=data_config[dataset_name]['data_dir'], 
                                   labels=data_config[dataset_name]['labels'],
                                   val_pct=val_pct, test_pct=0.0)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_image_data(image_sets, 
            input_shape=data_config[dataset_name]['input_shape'])
    
    debug()
    
    # Generate the model instance
    model = get_vgg16_model(use_relu)(
            input_shape=data_config[dataset_name]['input_shape'], classes=classes,
            weights=None, optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info("Print model summary:")
    model.summary()  # print model summary
    
    checkpt = gen_model_checkpoint()
    early_stopping = gen_early_stopping()
    
    history = model.fit(x=X_train, y=y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping, checkpt])

    logger.info("Successfully fit the model")
    
    model.save(c.MODELSTATE_DIR + '/' + c.MODEL_FILENAME)
    logger.info("Successfully saved the model")

    # Save metrics from the training process for later visualization
    save_train_metrics(history, metrics, filename=c.MODELSTATE_DIR + '/plot.txt')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ddsm')
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help='Percentage of total data as validation set')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--loss', default='categorical_crossentropy')

    args = parser.parse_args()

    main(**vars(args))
    logger.info('ALL DOE\n')
