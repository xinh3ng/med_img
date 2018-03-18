# -*- coding: utf-8 -*-
"""
  batch_size: No of samples per gradient update. Keras default is 32
"""

"""
  Key is data_src: data source name
"""
model_configs = {
    'ddsm': {
        'batch_size': 24
    },
    'mias': {
        'batch_size': 24  # 24
    },
    'mnist': {
        'batch_size': 32
    },
    'cifar10': {
        'batch_size': 32
    }
}
