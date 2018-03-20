# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import keras
from pydsutils.generic import create_logger
import med_img.mammo.utils.data_utils_ddsm as ddsm
import med_img.mammo.utils.data_utils_mias as mias
import med_img.mammo.utils.data_utils_mnist as mnist

debuglevel = 'info'
logger = create_logger(__name__, level=debuglevel)
seed = 1


def gen_img_sets_table(data_src: str, val_pct: float, test_pct: float,
                       verbose: int = 0) -> pd.DataFrame:
    """A factory function that creates image set table depending on data source

    Args:
        data_src:
        val_pct:
        test_pct:
    """
    df = fn_map[data_src]['create_img_sets'](val_pct, test_pct, verbose=verbose)
    assert all([(x in df.columns) for x in ['filename', 'label', 'label_num', 'type']]), \
        'Missing requreed columns'
    return df


def batch_gen_model_data(data_src, img_sets_table, type, sample_sizes, input_shape,
                         n_jobs=1, batch_size=0):
    """Factory function that create a model data iterator

    Args:
        data_src:
        img_sets_table:
        type: train or val or test
        input_shape:
        batch_size:
    Returns:
    """
    sample_size = sample_sizes[0]  # default to be 'train'
    if type == 'val':
        sample_size = sample_sizes[1]
    elif type == 'test':
        sample_size = sample_sizes[2]

    data_tuple = fn_map[data_src]['gen_model_data'](
        data_src=data_src,
        img_sets_table=img_sets_table,
        type=type,
        sample_size=sample_size,
        input_shape=input_shape,
        n_jobs=n_jobs,
        batch_size=batch_size)
    if debuglevel == 'debug': list(data_tuple)  # NB xheng: turn on debug mode
    return data_tuple


def batch_gen_model_data_from_files(data_src, img_sets_table, type, sample_size, input_shape,
                                    n_jobs=1, batch_size=0):
    """From image file names to numpy array, reshape them accordingly

    X's shape should be (num_samples, height, width, channel)
    Args:
        data_src Data source, e.g. ddsm, mias. It decides location of image folder too.
        type: 'train' or 'val' or 'test'
    """
    data = img_sets_table[img_sets_table.type == type].reset_index(drop=True)

    # 3 scenarios: 1) data is empty -> return None
    #   2) batch_size = 0 -> return all data
    #   3) batch_size > 0 -> create an iterator that returns data chunk by chunk
    if len(data) == 0:  # if found no data
        return None, None
    if batch_size == 0:  # 0 means getting all images
        X = files_to_img_array(data_src, data['filename'], input_shape=input_shape, verbose=2)
        y = keras.utils.to_categorical(data['label_num'].values, len(set(data['label_num'])))
        logger.info('Shape of X for %s is: %s' % (type, str(X.shape)))
        logger.info('Shape of y for %s is: %s' % (type, str(y.shape)))
        return X, y

    def yield_packet():
        """This is the way to have return and yield in the same function
        """
        # NB: per Keras, generator must loop over its data indefinitely
        while True:
            cnt = 0
            # data = data.sample(frac=1, random_state=seed, axis=1)  # Reshuffling data
            for b in range(0, len(data), batch_size):
                end = min(len(data), b + batch_size)
                chunk = data.loc[b:(end-1)]
                cnt += len(chunk)
                # Convert a list of image files to a 4D numpy array
                X = files_to_img_array(data_src, chunk['filename'], input_shape, n_jobs=n_jobs,
                                       verbose=1)
                y = keras.utils.to_categorical(chunk['label_num'].values,
                                               len(set(chunk['label_num'])))
                assert len(set(chunk['label_num'].values)) > 1, 'Label data must have multiple classes'
                if end == len(data):
                    logger.info('batch_gen_model_data_from_files finished 1 cycle of all %s samples' % type)
                yield X, y
                if cnt >= sample_size:
                    logger.info('batch_gen_model_data_from_files reached desired sample_size. Early exiting')
                    break
    return yield_packet()


def files_to_img_array(data_src, filenames, input_shape, n_jobs=-1, verbose=0):
    """Convert (in parallel) a list of file names to numpy array
    Args:
        data_src: Data source, e.g. mias, ddsm. It also decides where the image folder is.
    Returns
        A numpy array with shape (num_samples, input_shape)
    """
    file_to_array = fn_map[data_src]['file_to_array']  # function to convert an image to numpy array
    if verbose >= 2:
        logger.info('files_to_img_array: converting %d images to a Numpy array. '
                    'n_jobs = %d ...' % (len(filenames), n_jobs))
    input_shape_array = [input_shape for _ in range(len(filenames))]
    X = Parallel(n_jobs=n_jobs)(delayed(file_to_array)(fn, s) for fn, s in \
                                zip(filenames, input_shape_array))
    # Take this example: batch_size is 32. Then X is 32 arrays, each is (1024, 768, 1)
    # After this step, X's shape becomes (32, 1024, 768, 1)
    X = np.array(X)
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
    },
    'mnist': {
        'create_img_sets': mnist.create_img_sets,
        'gen_model_data': mnist.gen_model_data('mnist'),
        'file_to_array': None
    },
    'cifar10': {
        'create_img_sets': mnist.create_img_sets,
        'gen_model_data': mnist.gen_model_data('cifar10'),
        'file_to_array': None
    }
}
