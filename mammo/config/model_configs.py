# -*- coding: utf-8 -*-
"""

"""

"""
    Key is data_src: data source name
    batch_size: No of samples per fit_generator. For mnist and cifar, it is always 0, i.e. do not use generator.
        batch_size is no. of samples per gradient update. mnist or cifar will use Keras default, i.e. 32
"""
model_configs = {
    'ddsm': {
        'batch_size': 32
    },
    'mias': {
        'batch_size': 32  # 32
    },
    'mnist': {
        'batch_size': 0
    },
    'cifar10': {
        'batch_size': 0
    }
}
