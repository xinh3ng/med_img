# -*- coding: utf-8 -*-
"""

"""
from pdb import set_trace as debug
import re
from typing import Type
from pydsutils.generic import create_logger
from pymlearn.dl_data import BaseModelDataValidator, TfModelDataValidator, TorchModelDataValidator

from med_img.mammo.models.base_models import BaseModelOperator
from med_img.mammo.models.models import TfVgg16ModelOperator, TfSimpleCnnModelOperator,\
    TorchVgg16ModelOperator

logger = create_logger(__name__, level='info')


def select_model_operator(model_name: str) -> Type[BaseModelOperator]:
    """Factory function that selects the model operator, which is detailed in models.py
    """
    x = {'tfvgg16': TfVgg16ModelOperator,
         'tfcnn': TfSimpleCnnModelOperator,
         'torchvgg16': TorchVgg16ModelOperator
         }
    if model_name not in x.keys():
        raise KeyError('%s is not a known model name' %model_name)
    return x[model_name]


def select_model_data_validator(model_name: str, num_classes: int, num_rows: int,
    num_columns: int, num_channels: int) -> Type[BaseModelDataValidator]:
    
    x = {'tf': TfModelDataValidator,
         'torch': TorchModelDataValidator
         }
    # backend is either tf or torch
    backend_name = re.search('tf|torch', model_name).group(0)  # convert model name to backend_name
    validator = x.get(backend_name, BaseModelDataValidator)(num_classes, num_rows, num_columns, num_channels)
    return validator
