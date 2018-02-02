# -*- coding: utf-8 -*-
"""

Procedure:
  Invoke virtual env (Python 3.6)
  $ python train_mammo_model.py --dataset_name=mnist --model_name=tfcnn --optimizer=adam --loss=categorical_crossentropy
"""
from pdb import set_trace as debug
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pydsutils.generic import create_logger

import med_img.mammo.config.constants as c
from med_img.mammo.config.config_data import data_config
from med_img.mammo.utils.data_utils import load_image_data_fn
from med_img.mammo.models.model_operation import select_model_operator, select_model_data_validator

logger = create_logger(__name__, level='info')


def gen_model_checkpoint():
    """Create model checkpoint callback for model training"""
    return ModelCheckpoint(
            c.MODEL_STATE_DIR + '/{epoch:02d}' + c.MODEL_FILENAME,
            monitor='val_loss', save_best_only=False, save_weights_only=False,
            mode='auto', period=1, verbose=1)


def gen_early_stopping():
    """Generate early stopping callback"""
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
    input_shape = data_config[dataset_name]['input_shape']
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_image_data_fn(dataset_name)(
            image_dir=data_config[dataset_name]['data_dir'], 
            labels=data_config[dataset_name]['labels'],
            val_pct=val_pct,
            test_pct=0.0,
            input_shape=input_shape)
    
    # Generate the model instance
    classes = len(data_config[dataset_name]['labels'].keys())  # number of classes
    model_operator = select_model_operator(model_name)(
            input_shape=data_config[dataset_name]['input_shape'], classes=classes,
            weights=None, optimizer=optimizer, loss=loss, metrics=metrics)
    model = model_operator.create_model(verbose=1)
    
    checkpt = gen_model_checkpoint()
    early_stopping = gen_early_stopping()
    
    X_train, y_train = model_operator.process_X(X_train), model_operator.process_y(y_train)
    X_val, y_val = model_operator.process_X(X_val), model_operator.process_y(y_val)
    
    model_data_validator = select_model_data_validator(model_name, num_classes=classes,
        num_rows=input_shape[0], num_columns=input_shape[1], num_channels=input_shape[2])
    
    model_data_validator.validate_X(X_train)
    model_data_validator.validate_y(y_train)
    model_data_validator.validate_X(X_val)
    model_data_validator.validate_y(y_val)
    
    history = model.fit(x=X_train, y=y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping, checkpt])
    logger.info("Successfully fit the model")
    
    score = model.evaluate(X_val, y_val, verbose=0)
    logger.info('Validation loss: {}'.format(score[0]))
    logger.info('Validation accuracy: {}'.format(score[1]))
    
    model.save(c.MODEL_STATE_DIR + '/' + c.MODEL_FILENAME)
    logger.info("Successfully saved the model")

    # Save metrics from the training process for later visualization
    save_train_metrics(history, metrics, filename=c.MODEL_STATE_DIR + '/plot.txt')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='mias',
                        help='Data source: mnist or mias or ddsm')
    parser.add_argument('--model_name', default='tfcnn',
                        help='Clf model: tfcnn or tfvgg16 or torchcnn')
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help='Pct of data as validation set')
    parser.add_argument('--optimizer', default='adam',
                        help='Optimizer algorithm: adam')
    parser.add_argument('--loss', default='categorical_crossentropy',
            help='Loss function used by optimizer: categorical_crossentropy or binary_crossentropy')

    args = parser.parse_args()
    main(**vars(args))
    logger.info('ALL DONE\n')
