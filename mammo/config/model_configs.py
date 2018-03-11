# -*- coding: utf-8 -*-
"""

  batch_size: No of samples per gradient update. Keras default is 32
"""

# batch_size:
model_configs = {
    'ddsm': {
        'batch_size': 32
    },
    'mias': {
        'batch_size': 24
    },
    'mnist': {
        'batch_size': 32
    },
    'cifar10': {
        'batch_size': 32
    }
}
