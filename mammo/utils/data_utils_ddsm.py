# -*- coding: utf-8 -*-
"""

RESOURCES:
  http://numpy-discussion.scipy.narkive.com/gIuuTTuR/load-movie-frames-in-python

"""
from pdb import set_trace as debug
from typing import Any
import os
import re
import glob
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from pydsutils.generic import create_logger
from med_img.mammo.config import data_configs as dc

pd.options.display.max_colwidth = 144
logger = create_logger(__name__)
img_dir = dc.src_data_configs['ddsm']['data_dir']
labels = dc.src_data_configs['ddsm']['labels']
seed = 1


def create_img_sets(val_pct, test_pct, extension='png', verbose=0):
    """Builds a list of train, validation and test images
    
    Analyzes the sub folders in the image directory, splits them by train / val / test and returns a data 
    structure describing the lists of images for each label.
    Args:
        val_pct: Percentage of images reserved for validation.
        test_pct: Percentage of the images to reserve for tests.
        extension: Image file extention. Default is 'LJPEG'
    """
    logger.info('Start to create ddsm image sets...')
    assert os.path.isdir(img_dir), "img_dir: %s not valid" % img_dir
    
    img_sets = pd.DataFrame()
    for filename in glob.iglob(img_dir + '/**/*.' + extension, recursive=True):
        x = np.random.uniform(low=0.0, high=1.0, size=1)
        if x[0] < test_pct:
            type = 'test'
        elif x[0] < test_pct + val_pct:
            type = 'val'
        else:
            type = 'train'

        label = filename_to_label(filename, labels.keys())
        row = pd.DataFrame([{'type': type, 'label': label,
                             'label_num': labels[label],  # numerical label
                             'filename': filename}])
        img_sets = pd.concat([img_sets, row], axis=0)

    img_sets = img_sets.sample(frac=1, random_state=seed).reset_index(drop=True)  # Reshuffle
    logger.info('Successfully created image sets')
    if verbose >= 1:
        logger.info('Showing 10 random examples:\n%s' %img_sets.sample(n=10).to_string(line_width=120))
    return img_sets


def filename_to_label(filename, labels):
    """Convert the label from the long filename
    """
    found = [(re.search(r'%s' % x, filename) is not None) for x in labels]
    assert sum(found) == 1, 'should find 1 and only 1 label, but it is not this way'
    label = [i for (i, v) in zip(labels, found) if v][0]
    return label


def file_to_array(filename, input_shape):
    """Helper function to Convert a single file to numpy array

    """
    img = load_img(filename, target_size=(input_shape[0], input_shape[1]))
    if input_shape[2] == 1:  # If need grayscale
        img = img.convert('L')
    img = img_to_array(img)
    return img
