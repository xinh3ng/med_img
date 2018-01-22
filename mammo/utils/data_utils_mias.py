# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import os
import numpy as np
import pandas as pd
import cv2

from med_img.mammo.utils.generic_utils import create_logger

pd.options.display.max_colwidth = 144
logger = create_logger(__name__)
seed = 0
np.random.seed(seed)

    
def create_image_sets(image_dir, labels, val_pct, test_pct, extension='pgm'):
    """Builds a list of training images from the file system
    
    Analyzes the sub folders in the image directory, splits them by train / val / test and returns a data 
    structure describing the lists of images for each label.
    
    Args:
        image_dir: A folder containing subfolders of images.
        labels: True label of the image given by the long filename
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
    image_sets.loc[image_sets.label == 'normal', 'label_num'] = 0.0
    
    # Create train / val label
    image_sets['name'] = 'train'
    indices = np.random.randint(low=0, high=len(image_sets.index),
                      size=int(len(image_sets.index) * val_pct))
    image_sets.loc[indices, 'name'] = 'val'
    
    image_sets['filename'] = image_dir + '/' + image_sets['filename_short'] + '.' + extension
    image_sets.reset_index(drop=True, inplace=True)
    logger.info('Successfully created image sets. Example:\n%s' % image_sets.head(5).to_string(line_width=144))
    return image_sets


def load_image_data(image_dir, labels, val_pct, test_pct,
                    input_shape):
    """Load the images as numpy arrays, reshape them accordingly
    
    X's shape should be (num_samples, height, width, channel)
    """
    image_sets = create_image_sets(image_dir, labels, val_pct, test_pct, extension='pgm')
    
    X, y = {}, {}
    for name in ['train', 'val', 'test']:
        filtered = image_sets[image_sets.name == name]
        if len(filtered) == 0:  # if found no data
            y[name], X[name] = None, None
        else:
            # Convert list of image files to a 4D numpy array
            X[name] = _files_to_image_arrays(filtered['filename'].values, input_shape=input_shape)
            y[name] = filtered['label_num'].values  # 
    
    logger.info("Successfully loaded image files as numpy arrays. Shape of X_train and y_train are: %s, %s"\
                % (str(X['train'].shape), str(y['train'].shape)))
    logger.info("Shape of X_val and y_val are: %s, %s" % (str(X['val'].shape), str(y['val'].shape)))
    return (X['train'], y['train']), (X['val'], y['val']), (X['test'], y['test']) 


def _files_to_image_arrays(filenames, input_shape):
    """Convert a list of filenames to image (numpy) arrays
    
    X's shape should be (num_samples, input_shape)
    """
    X = np.zeros((len(filenames),) + input_shape)
    idx = 0
    for fn in filenames:
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (input_shape[0], input_shape[1]))
        img = img[..., np.newaxis]
        
        #new_img = np.zeros(input_shape)
        #new_img = cv2.normalize(img, new_img, 0, 255, cv2.NORM_MINMAX)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # new_img = new_img.astype('float32')
        
        # cv2.imshow("Show by CV2",  new_img); cv2.waitKey()
        new_img = img
        X[idx, ...] = new_img
        idx += 1

    return X
