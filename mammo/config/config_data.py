
import os

data_config = {
    'ddsm': {
        'labels': ['benign', 'normal', 'malevolent'],
        'data_dir': '{}/data/DDSM/figment.csee.usf.edu/pub/DDSM/cases/'.format(os.environ['HOME']),
        'input_shape': (224, 224, 3),
        'model_file': 'trained.h5'
        }
    }
