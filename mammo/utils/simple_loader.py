# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import os
import subprocess
from io import StringIO
import numpy as np
import pandas as pd
import keras.preprocessing.image as kimage
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from med_img.mammo.config import constants as c

def rebalance_data(data, column='cancer', negative_ratio=2.0, seed=None):
    """Rebalance data  to desired positive / negative ratio if there are enough samples."""
    
    positives = data[data[column] == 1]
    negatives = data[data[column] == 0]   
    num_negatives = int(negative_ratio * len(positives.index))
    assert len(negatives.index) >= num_negatives
    
    # Downsample the negative samples
    balanced = pd.concat([positives, 
                          negatives.sample(n=num_negatives, inplace=True, random_state=seed)])
    # Reshuffle so that positive samples are not all together
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


class BatchGeneratorCreator(object):
    """Abstract base class used for batch loading image files.
    
    Batch loading is required because there could be 100s of 1000s of images.
    """
    def __init__(self, imgs_dir, ycol='cancer', 
                 validation_split=0.25, batch_size=500):
        self.image_generator = ImageDataGenerator(width_shift_range=0.1,
                                                  height_shift_range=0.1,
                                                  shear_range=0.05,
                                                  zoom_range=[0.05, 0.1],
                                                  horizontal_flip=True,
                                                  vertical_flip=True)
        # imgs_dir should be the path to directory containing the image files, including a trailing slash.
        self.imgs_dir = imgs_dir
        # Proportion of samples to use for validation.
        self.validation_split = validation_split
        self.batch_size = batch_size
        
        self.img_metadata = pd.read_csv(c.IMAGES_CROSSWALK_FILE, sep='\t',
                                        na_values='.')
        self.img_metadata.subjectId = self.img_metadata.subjectId.astype(str)
        if 'examIndex' in self.img_metadata:
            self.img_metadata.examIndex = self.img_metadata.examIndex.astype(int)

        # Set placeholder value of 0 for cancer status of all rows. This will be refilled
        self.img_metadata[self.ycol] = 0

        # This file is not present for the scoring docker image but it's
        # not needed since we don't need to know the cancer status of a
        # mammogram for the scoring phase.
        if 'examIndex' in self.img_metadata and os.path.exists(c.EXAMS_METADATA_FILE):
            # Need to find the cancer labels from the exams metadata
            exam_fields = ['subjectId', 'examIndex', 'cancerL', 'cancerR']
            ex = pd.read_csv(c.EXAMS_METADATA_FILE, sep='\t',
                             na_values='.', usecols=exam_fields)
            ex.cancerL = ex.cancerL.fillna(0)
            ex.cancerR = ex.cancerR.fillna(0)
            ex.subjectId = ex.subjectId.astype(str)
            ex.examIndex = ex.examIndex.astype(int)
            ex.cancerL = ex.cancerL.astype(int)
            ex.cancerR = ex.cancerR.astype(int)

            for index, img_row in self.img_metadata.iterrows():
                exams_row = ex.loc[(ex.subjectId == img_row['subjectId']) &
                                   (ex.examIndex == img_row['examIndex']),
                                   'cancerL':'cancerR']

                if exams_row.iloc[0]['cancerL'] == 1 and img_row['laterality'] == 'L':
                    self.img_metadata.set_value(index, self.ycol, 1)
                elif exams_row.iloc[0]['cancerR'] == 1 and img_row['laterality'] == 'R':
                    self.img_metadata.set_value(index, self.ycol, 1)

        self.training_metadata = self.img_metadata[:self.total_training_samples()]
        self.validation_metadata = self.img_metadata[self.total_training_samples():]

    def total_samples(self):
        return len(self.img_metadata.index)

    def total_training_samples(self):
        return int(self.total_samples() * (1 - self.validation_split))

    def total_validation_samples(self):
        return self.total_samples() - self.total_training_samples()

    def get_dataset(self, dataset):
        if not isinstance(dataset, str):
            return dataset

        if dataset == 'all':
            return self.img_metadata
        elif dataset == 'training':
            return self.training_metadata
        elif dataset == 'validation':
            return self.validation_metadata

    def get_generator(self, dataset='all', train_mode=True):
        metadata_frame = self.get_dataset(dataset)

        curr_idx = 0
        dataset_len = len(metadata_frame.index)

        # A batch cannot be larger than the total number of samples in the dataset.
        if not train_mode and self.batch_size > dataset_len:
            raise ValueError('Batch size {} is larger than the number of samples {} in the dataset.'
                             .format(self.batch_size, dataset_len))

        # Keras expects the generator to output samples infinitely.
        # Ideally we'll choose a number of samples to train on so we don't wrap around but
        # we still need to handle that case.
        while 1:
            num_added = 0
            x = np.empty((self.batch_size, c.WIDTH, c.HEIGHT, 3))
            if train_mode:
                y = np.empty(self.batch_size)

            # Need to ensure that we wrap around when we reach the end.
            wraparound_idx = -1
            end_idx = curr_idx + self.batch_size
            if end_idx > dataset_len:
                wraparound_idx = end_idx - dataset_len
                end_idx = dataset_len

            # Read in dcm files and load the converted pixel values into the x array.
            # Load the cancer labels into the y array.
            iter_range = range(curr_idx, end_idx) + range(0, wraparound_idx)
            for i in iter_range:
                row = metadata_frame.iloc[i]
                img_file = self.get_image_data(self.imgs_dir + row['filename'])
                x[num_added] = kimage.img_to_array(self.proc_img_data(img_file))

                if train_mode:
                    y[num_added] = int(row[self.ycol])
                curr_idx += 1
                num_added += 1

                if curr_idx >= dataset_len:
                    curr_idx = 0

            if train_mode:  # Generate extra images only when training
                self.image_generator.fit(x)
                x, y = self.image_generator.flow(x, y, batch_size=self.batch_size).next()
                yield x, y
            else:
                yield x

    def get_image_data(self, filename):
        """Override this"""
        raise NotImplementedError

    def proc_img_data(self, img):
        return img


class PNGBatchGeneratorCreator(BatchGeneratorCreator):
    """Used for batch loading 8-bit RGB PNG files."""

    def get_image_data(self, filename):
        real_name = filename.replace('.dcm', '.png')
        return Image.open(real_name)


class DICOMBatchGeneratorCreator(BatchGeneratorCreator):
    """Used for batch loading raw DICOM files."""
    def __init__(self, imgs_dir, validation_split=0.25, batch_size=500):
        super(DICOMBatchGeneratorCreator, self).__init__(imgs_dir, validation_split, batch_size)

        # Need to add /usr/local/bin to PATH for local testing on MacOS
        self._env = os.environ.copy()
        self._env['PATH'] = '/usr/local/bin:' + self._env['PATH']

    def get_image_data(self, filename):
        cmd = 'convert ' + filename + ' ' + c.IMAGE_MAGICK_FLAGS + ' png:-'
        raw_img = subprocess.check_output(cmd, env=self._env, shell=True)
        return Image.open(StringIO.StringIO(raw_img))

