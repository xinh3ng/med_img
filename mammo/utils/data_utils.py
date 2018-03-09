# -*- coding: utf-8 -*-
from pdb import set_trace as debug
from typing import Callable
import pandas as pd
import keras
from pydsutils.generic import create_logger
import med_img.mammo.utils.data_utils_ddsm as ddsm
import med_img.mammo.utils.data_utils_mias as mias
import med_img.mammo.utils.data_utils_mnist as mnist

debuglevel = 'info'
logger = create_logger(__name__, level=debuglevel)
# np.set_printoptions(precision=4)

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
                         batch_size=0, n_jobs=-1, verbose=0):
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
                        batch_size=batch_size,
                        n_jobs=n_jobs,
                        verbose=verbose)
    if debuglevel == 'debug': list(it)  # NB xheng: turn on debug mode
    return it


def batch_gen_model_data_from_files(data_src, img_sets_table, type, sample_size, input_shape,
                                    batch_size=0, n_jobs=-1, verbose=0):
    """From image file names to numpy array, reshape them accordingly

    X's shape should be (num_samples, height, width, channel)
    """
    assert batch_size >= 0
    files_to_img_array = fn_map[data_src]['files_to_img_array']

    data = img_sets_table[img_sets_table.type == type].reset_index(drop=True)
    cnt = 0

    # 3 scenarios: 1) data is empty -> return None
    #   2) batch_size = 0 -> return all data
    #   3) batch_size > 0 -> create an iterator that returns data chunk by chunk
    if len(data) == 0:  # if found no data
        return None, None

    if batch_size == 0:  # 0 means getting all images
        X = files_to_img_array(data['filename'], input_shape=input_shape)
        y = keras.utils.to_categorical(data['label_num'].values, len(set(data['label_num'])))
        yield X, y

    else:
        for b in range(0, len(data), batch_size):
            end = min(len(data), b + batch_size)
            chunk = data.loc[b:(end-1)]
            cnt += len(chunk)
            # Convert a list of image files to a 4D numpy array
            X = files_to_img_array(chunk['filename'], input_shape, n_jobs=n_jobs,
                                   verbose=verbose)
            y = keras.utils.to_categorical(chunk['label_num'].values, len(set(chunk['label_num'])))
            assert len(set(chunk['label_num'].values)) > 1, 'Label data must have multiple classes'
            yield X, y
            if cnt >= sample_size:  # If reaching desired sample size
                break


"""A suite functions being called during data acquisition
"""
fn_map = {
    'mias': {
        'create_img_sets': mias.create_img_sets,
        'gen_model_data': batch_gen_model_data_from_files,
        'files_to_img_array': mias.files_to_img_array
    },
    'ddsm': None
}



