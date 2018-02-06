# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import os
import re
import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pydsutils.generic import create_logger

pd.options.display.max_colwidth = 144
logger = create_logger(__name__)
seed = 0
np.random.seed(seed)


def filename_to_label(filename, labels):
    """Pick out the label from the long filename"""
    
    found = [(re.search(r'%s' % x, filename) is not None) for x in labels]
    assert sum(found) == 1, 'should find 1 and only 1 label, but it is not this way'
    label = [i for (i, v) in zip(labels, found) if v][0]
    return label
    
    
def create_image_sets(image_dir, labels, val_pct, test_pct, extension='LJPEG'):
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
    
    image_sets = pd.DataFrame()
    for filename in glob.iglob(image_dir + '/**/*.' + extension, recursive=True):
        
        x = np.random.uniform(low=0.0, high=1.0, size=1)
        if x[0] < val_pct:
            name = 'val'
        elif x[0] < test_pct + val_pct:
            name = 'test'
        else:
            name = 'train'
        
        label = filename_to_label(filename, labels.keys())
        row = pd.DataFrame([{'name': name, 'label': label, 'label_num': labels[label],  # numerical label
                             'filename': filename}])
        image_sets = pd.concat([image_sets, row], axis=0)
    image_sets.reset_index(drop=True, inplace=True)
    logger.info('Successfully created image sets. Example: %s' % image_sets.head(5).to_string(line_width=144))
    return image_sets


def load_image_data(image_sets, input_shape=(50, 50, 3)):
    """Load the images as numpy arrays, reshape them accordingly
    
    X's shape should be (num_samples, height, width, channel)
    """
    
    X, y = {}, {}
    for name in ['train', 'val', 'test']:
        filtered = image_sets[image_sets.name == name]
        if len(filtered.index) == 0:  # if found no data
            y[name], X[name] = None, None
        else:
            y[name] = filtered['label_num'].values
            # Convert list of image files to a 4D numpy array
            X[name] = _files_to_image_arrays(filtered['filename'].values, input_shape=input_shape)
    
    return (X['train'], y['train']), (X['val'], y['val']), (X['test'], y['test']) 


def _files_to_image_arrays(filenames, input_shape):
    """Convert a list of filenames to image (numpy) arrays
    
    X's shape should be (num_samples, input_shape)
    """
    X = []
    for fn in filenames:
        debug()
        img = Image.open(fn)
        img.show()
        img = cv2.imread(fn)
        img = cv2.resize(img, input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(image).reshape((3,224,224))
        img = img.astype('float32')
        X.append(img)
    assert X.shape == (len(filenames), 1) + input_shape, "X's shape is not (num_samples, height, width, channel)"
    return X



