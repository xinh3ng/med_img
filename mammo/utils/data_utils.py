# -*- coding: utf-8 -*-
from pdb import set_trace as debug
from typing import Callable

import med_img.mammo.utils.data_utils_ddsm as ddsm
import med_img.mammo.utils.data_utils_mias as mias
import med_img.mammo.utils.data_utils_mnist as mnist


def load_image_data_fn(dataset_name: str) -> Callable:
    """Select which image data source to convert to numpy arrays (for model training"""
    x = {'ddsm': ddsm.load_image_data,
         'mias': mias.load_image_data,
         'mnist': mnist.load_image_data
         }
    if dataset_name not in x.keys():
        raise KeyError('{} not found'.format(dataset_name))
    return x[dataset_name]
