# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import os
from typing import Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import keras
from keras.preprocessing.image import load_img, img_to_array
from pydsutils.generic import create_logger

from med_img.mammo.config import data_config as dc

pd.options.display.max_colwidth = 144
logger = create_logger(__name__)
image_dir = dc.src_data_configs['mias']['data_dir']
labels = dc.src_data_configs['mias']['labels']
NJOBS = -1  # for multi processing


def load_image_data(val_pct, test_pct, input_shape,
                    *args: Any, **kwargs: Any):
    """Main function: Load the images as numpy arrays, reshape them accordingly

    X's shape should be (num_samples, height, width, channel)
    """
    image_sets = create_image_sets(val_pct, test_pct, extension='pgm')

    X, y = {}, {}
    for name in ['train', 'val', 'test']:
        filtered = image_sets[image_sets.name == name]
        if len(filtered) == 0:  # if found no data
            y[name], X[name] = None, None
        else:
            # Convert list of image files to a 4D numpy array
            X[name] = files_to_image_arrays(filtered['filename'].values, input_shape=input_shape)
            y[name] = keras.utils.to_categorical(filtered['label_num'].values, len(labels))

    return (X['train'], y['train']), (X['val'], y['val']), (X['test'], y['test'])


def create_image_sets(val_pct, test_pct, extension='pgm'):
    """Builds a list of training images from the file system
    
    Analyzes the sub folders in the image directory, splits them by train / val / test and returns a data 
    structure describing the lists of images for each label.
    
    Args:
        test_pct: Percentage of the images to reserve for tests.
        val_pct: Percentage of images reserved for validation.
    """
    assert os.path.isdir(image_dir), "image_dir: %s not valid" % image_dir
    
    # Load image information from true_labels.txt
    image_sets = pd.read_csv(image_dir + '/true_labels.txt', sep=' ')
    is_duped = image_sets.duplicated('filename_short')  # Remove duplicated
    image_sets = image_sets[~is_duped].reset_index(drop=True)
    
    # Add label and label_num columns
    image_sets['label'] = 'cancer'
    image_sets.loc[image_sets.label_long == 'NORM', 'label'] = 'normal'    
    image_sets['label_num'] = 1.0  # label in numerical format
    image_sets.loc[image_sets.label == 'normal', 'label_num'] = 0.0  # normal = 0
    
    # Create train / val label
    image_sets['name'] = 'train'
    indices = np.random.randint(low=0, high=len(image_sets.index),
                      size=int(len(image_sets.index) * val_pct))
    image_sets.loc[indices, 'name'] = 'val'
    
    image_sets['filename'] = image_dir + '/' + image_sets['filename_short'] + '.' + extension
    image_sets.reset_index(drop=True, inplace=True)
    logger.info('Successfully created image sets. Example:\n%s' % image_sets.head(5).to_string(line_width=144))
    return image_sets


def files_to_image_arrays(filenames, input_shape, n_jobs=NJOBS):
    """Convert a list of filenames to image (numpy) arrays
    
    X's shape should be (num_samples, input_shape)
    """    
    logger.info("Start to convert %d image files to numpy arrays" % len(filenames))        
    input_shape_array = [input_shape for _ in range(len(filenames))]
    X = Parallel(n_jobs=n_jobs)(delayed(file_to_array)(fn, s) for fn, s in \
                                zip(filenames, input_shape_array))
    X = np.array(X)
    logger.info("Successfully converted image files to numpy arrays")
    return X


def file_to_array(filename, input_shape):
    """Convert a single file to image (numpy array)

    """
    img = load_img(filename, target_size=(input_shape[0], input_shape[1]))  # PIL object
    img = img_to_array(img)

    #new_img = np.zeros(input_shape)
    #new_img = cv2.normalize(img, new_img, 0, 255, cv2.NORM_MINMAX)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # new_img = new_img.astype('float32')
    
    # cv2.imshow("Show by CV2",  new_img); cv2.waitKey()
    new_img = img
    return new_img
