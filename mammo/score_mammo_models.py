# -*- coding: utf-8 -*-
"""
Shows how to toss a capsule to a container.
"""
from pdb import set_trace as debug
import os
import pandas as pd
import keras.models as km
from pydsutils.generic import create_logger

import med_img.mammo.utils.constants as c
import med_img.mammo.utils.simple_loader as sl


logger  = create_logger(__name__, level='info')


if __name__ == '__main__':

    # Load up the pre-trained model
    model = km.load_model(c.MODELSTATE_DIR + '/' + c.MODEL_FILENAME)

    batch_size = 500
    creator = sl.DICOMBatchGeneratorCreator(c.INFERENCE_IMG_DIR + '/', batch_size=batch_size)
    generator = creator.get_generator('all', False)

    images_fields = ['subjectId', 'laterality']
    images_meta = pd.read_csv(c.IMAGES_CROSSWALK_FILEPATH, sep='\t', na_values='.', 
                              usecols=images_fields)
    images_meta.subjectId = images_meta.subjectId.astype(str)
    images_meta.laterality = images_meta.laterality.astype(str)

    # Need to temporarily keep multiple confidence scores for each subjectId and laterality as there
    # could be multiple mammograms per breast for a subject.
    results = pd.DataFrame(columns=['subjectId', 'laterality', 'confidence_sum', 'num_scores'])
    results.subjectId = results.subjectId.astype(str)
    results.laterality = results.laterality.astype(str)
    results.num_scores = results.num_scores.astype(int)
    curr_index = 0

    total_samples = creator.total_samples()
    assert len(images_meta.index) == total_samples

    # Continue looping until all samples have been predicted.
    while curr_index < total_samples:
        print("Currently finished scoring {} samples".format(curr_index))

        predicted_vals = model.predict_on_batch(generator.next())

        for val in predicted_vals:
            row = images_meta.iloc[curr_index]
            new_prediction = val[0]

            result_row = results[(results.subjectId == row['subjectId']) &
                                 (results.laterality == row['laterality'])]
            if result_row.empty:
                # Found new subject and laterality. Add a new row to the DataFrame
                new_row = pd.DataFrame([[row['subjectId'], row['laterality'], new_prediction, 1]],
                                       columns=['subjectId', 'laterality', 'confidence_sum', 'num_scores'])
                results = results.append(new_row, ignore_index=True)
            else:
                # Add a new prediction to existing subject and laterality.
                results.loc[(results.subjectId == row['subjectId']) &
                            (results.laterality == row['laterality']), 'confidence_sum'] += new_prediction
                results.loc[(results.subjectId == row['subjectId']) &
                            (results.laterality == row['laterality']), 'num_scores'] += 1

            curr_index += 1

            # If the total number of samples is not a multiple of the batch size
            # we want to exit early rather than reprocessing the data that wrapped around.
            if curr_index >= total_samples:
                break

    print("Finished scoring all {} samples".format(curr_index))

    # TODO:
    # Instead of just averaging the scores for each subject and laterality we could take
    # the max predicted score so we output 1 when at least one mammogram shows potential to show
    # breast cancer.
    results['confidence'] = results['confidence_sum'] / results['num_scores']
    print(results)

    # Ready to output the relevant columns to tsv file now.
    output_fields = ['subjectId', 'laterality', 'confidence']
    results.to_csv(c.OUTPUT_FILEPATH, sep='\t', columns=output_fields, index=False)
    logger.info('ALL DONE\n')
    