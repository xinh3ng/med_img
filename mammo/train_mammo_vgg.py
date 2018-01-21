# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

from med_img.mammo.utils.generic_utils import create_logger
import med_img.mammo.config.constants as c
from med_img.mammo.config.config_data import data_config
import med_img.mammo.utils.simple_loader as sl
from med_img.mammo.models.mammo_vgg import get_vgg16_model

logger = create_logger(__name__, level='info')
seed = os.getenv('RANDOM_SEED', 1337) # Must use the RANDOM_SEED environment as specified in challenge guidelines.


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


def main(dataset_name='ddsm', negative_ratio=1.0,
         use_relu=False, batch_size=32, epochs=25,
         optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    """Main function
    
    :param negative_ratio: Ratio of pos/neg samples. 2 means every 100 pos samples, we use 200 neg samples
    """
    np.random.seed(int(seed))

    classes = len(data_config[dataset_name])
   
    validation_split = 0.25
    blowup = 5  # Number of times the data in the balanced dataset should be augmented.
    #nb_filters = 32      # number of convolutional filters to use
    #pool_size = (2, 2)      # size of pooling area for max pooling
    #kernel_size = (3, 3)      # convolution kernel size
    
    # Generate model instance
    gen_vgg_fn = get_vgg16_model(use_relu)
    model = gen_vgg_fn(
            input_shape=data_config[dataset_name]['input_shape'], classes=classes,
            weights=None, optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info("Print model summary:")
    model.summary()  # print model summary
    
    # Get a balanced dataset with a negative:positive ratio
    creator = sl.PNGBatchGeneratorCreator(c.PREPROCESS_IMG_DIR,
                                          batch_size=batch_size, validation_split=validation_split)
    balanced = sl.rebalance_data(creator.get_dataset('training'), 
                                 negative_ratio=negative_ratio)

    # Number of samples per epoch must be a multiple of batch size. Thus we'll use the largest
    # multiple of batch size possible. This wastes at most batch size amount of samples.
    # Also limit training to 20000 images max due to time constraints.
    num_training_samples = min(15000, len(balanced.index) * blowup) // batch_size * batch_size
    num_validation_samples = num_training_samples * validation_split

    # Create callback functions during the long training process
    model_checkpoint = ModelCheckpoint(
            c.MODELSTATE_DIR + '/{epoch:02d}' + c.MODEL_FILENAME,
            monitor='val_loss', save_best_only=False, save_weights_only=False,
            mode='auto', period=1, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, 
                                   verbose=1, mode='auto')

    logger.info('Start training on set of {} images with a negative ratio of {}'\
                .format(num_training_samples, negative_ratio))
    history = model.fit_generator(
            creator.get_generator(dataset=balanced), 
            steps_per_epoch=num_training_samples, epochs=epochs, 
            validation_data=creator.get_generator('validation'), validation_steps=num_validation_samples,
            callbacks=[early_stopping, model_checkpoint])
    logger.info("Successfully fit the model")
    
    model.save(c.MODELSTATE_DIR + '/' + c.MODEL_FILENAME)
    logger.info("Successfully saved the model")

    # Save metrics from the training process for later visualization
    save_train_metrics(history, metrics, 
                       filename=c.MODELSTATE_DIR + '/plot.txt')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ddsm')
    parser.add_argument('--negative_ratio', type=float, default='1.0')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--loss', default='binary_crossentropy')

    args = parser.parse_args()

    main(**vars(args))
    logger.info('ALL DOE\n')
