# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import os
from typing import Any
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from pydsutils.generic import create_logger
from med_img.mammo.config import data_configs as dc

pd.options.display.max_colwidth = 120
logger = create_logger(__name__)
img_dir = dc.src_data_configs['mias']['data_dir']
labels = dc.src_data_configs['mias']['labels']
seed = 1


def create_img_sets(val_pct, test_pct=0.0, extension='pgm', verbose=0) -> pd.DataFrame:
    """Builds a list of train / val / test image filenames

    Analyzes the sub folders in the image directory, splits them by train / val / test and returns a data
    structure describing the lists of images for each label.
    The only useful output columns are: 'filename', 'label', 'label_num', 'type'

    Args:
        test_pct: Percentage of the images to reserve for tests.
        val_pct: Percentage of images reserved for validation.
    """
    assert os.path.isdir(img_dir), "img_dir: %s not valid" % img_dir

    # Load image information from true_labels.txt
    img_sets = pd.read_csv(img_dir + '/true_labels.txt', sep=' ')
    is_duped = img_sets.duplicated('filename_short')  # Remove duplicated
    img_sets = img_sets[~is_duped].reset_index(drop=True)

    # Add label and label_num columns
    img_sets['label'] = 'cancer'
    img_sets.loc[img_sets.label_long == 'NORM', 'label'] = 'normal'
    img_sets['label_num'] = 1.0  # label in numerical format
    img_sets.loc[img_sets.label == 'normal', 'label_num'] = 0.0  # normal = 0

    # Create train / val label
    img_sets['type'] = 'train'
    indices = np.random.randint(low=0, high=len(img_sets.index),
                                size=int(len(img_sets.index) * val_pct))
    img_sets.loc[indices, 'type'] = 'val'
    img_sets['filename'] = img_dir + '/' + img_sets['filename_short'] + '.' + extension
    img_sets = img_sets.sample(frac=1,  random_state=seed).reset_index(drop=True)  # Reshuffle

    logger.info('Successfully created image sets. Example:')
    if verbose >= 1: logger.info('Example:\n%s' % img_sets.head(5).to_string(line_width=144))
    return img_sets



def file_to_array(filename, input_shape):
    """Convert a single file to image (numpy array)

    """
    img = load_img(filename, target_size=(input_shape[0], input_shape[1]))  # PIL object
    if input_shape[2] == 1:  # If need grayscale
        img = img.convert('L')  # To grayscale
    img = img_to_array(img)
    return img
