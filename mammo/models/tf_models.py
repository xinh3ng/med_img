# -*- coding: utf-8 -*-
"""

MODELS:
  tfcnn
  tfvgg16

"""
from pdb import set_trace as debug
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input
from pydsutils.generic import create_logger
from med_img.mammo.models.base_models import BaseTfModel

logger = create_logger(__name__, level='info')
# tf_metrics = ['acc', 'mse', precision, recall]
tf_metrics = ['acc', 'mse']

#####################################################
# Goals include:
#   Preprocess input data: normalize, one-hot encoding, outliers, etc.
#   Create a model of choice
#####################################################

class TfVgg16(BaseTfModel):
    def __init__(self, input_shape, num_classes, batch_size,
                 include_top=True, weights=None,
                 optimizer='adam', loss='categorical_crossentropy'):
        super(TfVgg16, self).__init__( input_shape, num_classes, batch_size,
                                       include_top, weights,
                                       optimizer, loss)

    def create_model(self, verbose=0):
        """Instantiate the VGG16 architecture
    
        """
        model = VGG16(include_top=self.include_top, weights=self.weights,
                      input_shape=self.input_shape, pooling=None,
                      classes=self.num_classes)
        
        model.compile(optimizer=self.optimizer, loss=self.loss,
                      metrics=tf_metrics)
        logger.info("Successfully compiled the model")
        return model
    
    def process_X(self, X):
        return preprocess_input(X)  # for vgg16 only
    

class TfSimpleCnn(BaseTfModel):
    def __init__(self, input_shape, num_classes, batch_size,
                 include_top=True, weights=None,
                 optimizer='adam', loss='categorical_crossentropy'):
        super(TfSimpleCnn, self).__init__(input_shape, num_classes, batch_size,
                                          include_top, weights,
                                          optimizer, loss)

    def create_model(self, verbose=0):
        """Instantiate a simple Cnn model
        
        https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
        """    
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
    
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=tf_metrics)
        logger.info("Successfully compiled the model")
        return model
