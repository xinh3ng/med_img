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
from joblib import Parallel, delayed
import keras
from keras.preprocessing.image import load_img, img_to_array
from pydsutils.generic import create_logger
from med_img.mammo.config import data_config as dc

pd.options.display.max_colwidth = 144
logger = create_logger(__name__)
img_dir = dc.src_data_configs['ddsm']['data_dir']
labels = dc.src_data_configs['ddsm']['labels']
NJOBS = -1  # for multiprocessing


def load_img_data(val_pct, test_pct, input_shape, *args: Any, **kwargs: Any):
    """Main function: Load the images as numpy arrays, reshape them accordingly

    X's shape should be (num_samples, height, width, channel)
    """
    # Create a lot of train, validation, test images, e.g.
    #  filename    label    label_num   type
    #     0.png   normal            0  train 
    #     1.png   cancer            1    val
    img_sets = create_img_sets(val_pct, test_pct)

    X, y = {}, {}
    for type in ['train', 'val', 'test']:
        filtered = img_sets[img_sets.type == type]
        if len(filtered.index) == 0:  # if found no data
            y[type], X[type] = None, None
        else:
            # Convert list of image files to a 4D numpy array
            X[type] = files_to_img_array(filtered['filename'].values,
                                           input_shape=input_shape)
            y[type] = keras.utils.to_categorical(filtered['label_num'].values, len(labels))

    return (X['train'], y['train']), (X['val'], y['val']), (X['test'], y['test'])


def create_img_sets(val_pct, test_pct, extension='png'):
    """Builds a list of train, validation and test images
    
    Analyzes the sub folders in the image directory, splits them by train / val / test and returns a data 
    structure describing the lists of images for each label.
    Args:
        val_pct: Percentage of images reserved for validation.
        test_pct: Percentage of the images to reserve for tests.
        extension: Image file extention. Default is 'LJPEG'
    """
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
    img_sets.reset_index(drop=True, inplace=True)
    logger.info('Successfully created image sets. Showing random selected examples:\n%s' %\
            img_sets.sample(n=10).to_string(line_width=120))
    return img_sets


def filename_to_label(filename, labels):
    """Pick out the label from the long filename
    """
    found = [(re.search(r'%s' % x, filename) is not None) for x in labels]
    assert sum(found) == 1, 'should find 1 and only 1 label, but it is not this way'
    label = [i for (i, v) in zip(labels, found) if v][0]
    return label


def files_to_img_array(filenames, input_shape, n_jobs=NJOBS):
    """Convert a list of file names to an 4d numpy array

    Return:
        X: shape should be (num_samples, input_shape)
    """
    logger.info("Start to convert %d image files to a numpy array" % len(filenames))
    input_shape_array = [input_shape for _ in range(len(filenames))]
    X = Parallel(n_jobs=n_jobs)(delayed(file_to_array)(fn, s) for fn, s in \
                                zip(filenames, input_shape_array))
    X = np.array(X)
    logger.info("Successfully converted image files to a 4D numpy array")
    return X


def file_to_array(filename, input_shape):
    """Convert a single file to image (numpy array)

    """
    # It is grayscale image
    img = load_img(filename, target_size=(input_shape[0], input_shape[1]))
    img = img_to_array(img)
    return img

