import os
import pandas as pd
from pathlib import Path
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('path_to_dataset', None, 'Full path to unzpipped 2017-IWT4S-HDR_LP-dataset.')
flags.mark_flag_as_required('path_to_dataset')


def labels_to_images(path_to_dataset):
    """
    converting 2017-IWT4S-HDR_LP-dataset images to handy format where each image equals it ground truth label
    :param path_to_dataset: full path to unzpipped 2017-IWT4S-HDR_LP-dataset
    :return: stores result in /image subfolder of your path_to_dataset param
    """
    csv_file = path_to_dataset + 'trainVal.csv'
    df = pd.read_csv(csv_file)
    # print(df, df.shape)
    dst_folder = path_to_dataset+'images'
    os.makedirs(dst_folder, exist_ok=True)
    for i in range(df.shape[0]):
        each = df[i:i+1]
        lp = each.lp.values[0]
        image_path = each.image_path.values[0]
        # print(path_to_dataset + image_path[2:], dst_folder+'/'+lp+'_'+str(i)+'.jpg')
        os.rename(path_to_dataset + image_path[2:], dst_folder+'/'+lp+'_'+str(i)+'.png')


def main(args):
    path_to_dataset = FLAGS.path_to_dataset  # '/mnt/mongo-data/downloads/2017-IWT4S-HDR_LP-dataset'
    if path_to_dataset[-1] != '/':
        path_to_dataset += '/'
    try:
        labels_to_images(path_to_dataset)
    except:
        logging.error('Path_to_dataset is incorrect')


if __name__ == '__main__':
    app.run(main)


