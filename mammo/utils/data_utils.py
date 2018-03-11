# -*- coding: utf-8 -*-
from pdb import set_trace as debug
from typing import Callable
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import keras
from pydsutils.generic import create_logger
import med_img.mammo.utils.data_utils_ddsm as ddsm
import med_img.mammo.utils.data_utils_mias as mias

debuglevel = 'info'
logger = create_logger(__name__, level=debuglevel)
# np.set_printoptions(precision=4)
seed = 1

__all__ = ['gen_img_sets_table', 'batch_gen_model_data', 'batch_gen_model_data_from_files',
           'fn_map']

def gen_img_sets_table(data_src: str, val_pct: float, test_pct: float,
                       verbose: int=0) -> pd.DataFrame:
    """A factory function that creates image set table depending on data source

    :param data_src:
    :param val_pct:
    :param test_pct:
    :return:
    """
    create_img_sets_fn = fn_map[data_src]['create_img_sets']
    df = create_img_sets_fn(val_pct, test_pct, verbose=verbose)
    assert all([(x in df.columns) for x in ['filename', 'label', 'label_num', 'type']]),\
        'Missing requreed columns'
    return df

def report_img_sets_stats(img_sets_table):

    train_length = sum(img_sets_table['type'] == 'train')
    val_length = sum(img_sets_table['type'] == 'val')
    logger.info('Total train and val data lengths: %d, %d' %(train_length, val_length))

    for label in set(img_sets_table['label']):
        logger.info('Total length of %s label is: %d' % (label, sum(img_sets_table['label'] == label)))
    return


def batch_gen_model_data(data_src, img_sets_table, type, sample_sizes, input_shape,
                         n_jobs=1, batch_size=0, verbose=0):
    """Factory function that create a model data iterator

    Args:
        data_src:
        img_sets_table:
        type: train or val or test
        input_shape:
        batch_size:
    Returns:
    """
    if type == 'train':
        sample_size = sample_sizes[0]
    elif type == 'val':
        sample_size = sample_sizes[1]
    else:
        sample_size = sample_sizes[2]

    gen_model_data = fn_map[data_src]['gen_model_data']
    it = gen_model_data(data_src=data_src,
                        img_sets_table=img_sets_table,
                        type=type,
                        sample_size=sample_size,
                        input_shape=input_shape,
                        n_jobs=n_jobs,
                        batch_size=batch_size,
                        verbose=verbose)
    if debuglevel == 'debug': list(it)  # NB xheng: turn on debug mode
    return it


def batch_gen_model_data_from_files(data_src, img_sets_table, type, sample_size, input_shape,
                                    n_jobs=1, batch_size=0, verbose=0):
    """From image file names to numpy array, reshape them accordingly

    X's shape should be (num_samples, height, width, channel)
    """
    assert batch_size >= 0
    data = img_sets_table[img_sets_table.type == type].reset_index(drop=True)
    cnt = 0

    # 3 scenarios: 1) data is empty -> return None
    #   2) batch_size = 0 -> return all data
    #   3) batch_size > 0 -> create an iterator that returns data chunk by chunk
    if len(data) == 0:  # if found no data
        return None, None

    if batch_size == 0:  # 0 means getting all images
        X = files_to_img_array(data_src, data['filename'], input_shape=input_shape)
        y = keras.utils.to_categorical(data['label_num'].values, len(set(data['label_num'])))
        return X, y

    # NB: generator expected to loop over its data indefinitely according to Keras
    while True:
        # data = data.sample(frac=1, random_state=seed, axis=1)  # Reshuffling data
        for b in range(0, len(data), batch_size):
            end = min(len(data), b + batch_size)
            chunk = data.loc[b:(end-1)]
            cnt += len(chunk)
            # Convert a list of image files to a 4D numpy array
            X = files_to_img_array(data_src, chunk['filename'], input_shape, n_jobs=n_jobs,
                                   verbose=verbose)
            y = keras.utils.to_categorical(chunk['label_num'].values, len(set(chunk['label_num'])))
            assert len(set(chunk['label_num'].values)) > 1, 'Label data must have multiple classes'
            yield X, y
            if cnt >= sample_size:  # If reaching desired sample size
                break


def files_to_img_array(data_src, filenames, input_shape, n_jobs=-1, verbose=0):
    """Convert (in parallel) a list of file names to numpy array
    Args:
        data_src: Data source, e.g. mias, ddsm
    Returns
        A numpy array with shape (num_samples, input_shape)
    """
    file_to_array = fn_map[data_src]['file_to_array']  # function to convert an image to numpy array
    if verbose >= 2:
        logger.info("Start to convert %d image files to numpy arrays" % len(filenames))
    input_shape_array = [input_shape for _ in range(len(filenames))]
    X = Parallel(n_jobs=n_jobs)(delayed(file_to_array)(fn, s) for fn, s in \
                                zip(filenames, input_shape_array))
    X = np.array(X)
    if verbose >= 2: logger.info("Successfully converted image files to numpy arrays")
    return X


"""A suite functions being called during data acquisition
"""
fn_map = {
    'mias': {
        'create_img_sets': mias.create_img_sets,
        'gen_model_data': batch_gen_model_data_from_files,
        'file_to_array': mias.file_to_array
    },
    'ddsm': {
        'create_img_sets': ddsm.create_img_sets,
        'gen_model_data': batch_gen_model_data_from_files,
        'file_to_array': ddsm.file_to_array

    }
}



