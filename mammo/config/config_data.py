
import os

data_config = {
    'ddsm': {
        'labels': {'normal': 0, 'benign': 1, 'cancer': 2},
        'data_dir': '{}/data/DDSM/figment.csee.usf.edu/pub/DDSM/cases/'.format(os.environ['HOME']),
        'input_shape': (50, 50, 3) # (224, 224, 3)
        },
    'mias': {
        'labels': {'NORMAL': 0, 'CANCER': 1},
        'data_dir': '{}/data/mias/'.format(os.environ['HOME']),
        'input_shape': (224, 224, 1) # (224, 224, 1)
        },
    
    }
