import __init__

import sys
import os
import glob

import tensorflow as tf
import numpy as np

from pose_3d.data_helpers import read_maps_poses_images

DATASET_PATH = '/mnt/Data/ben/surreal/SURREAL/data/cmu/train/run0/'
SUMMARY_DIR = '/home/ben/tensorflow_logs/3d_pose'
SAVER_PATH = '/home/ben/tensorflow_logs/3d_pose/ckpts/3d_pose.ckpt'
    
if __name__ == '__main__':
    dataset_dir = os.path.realpath(DATASET_PATH)
    basenames = os.listdir(dataset_dir)

    maps_files = []
    info_files = []
    frames_paths = []
    for basename in basenames:
        one_data_dir = os.path.join(dataset_dir, basename)
        one_dir_maps_files = glob.glob(
            os.path.join(one_data_dir, basename + '_c*_maps.mat'))
        # only get the info file and frames for heatmaps that exist
        one_dir_info_files = map(lambda fn: fn[:-9] + '_info.mat', 
                                 one_dir_maps_files)
        one_dir_frames_paths = map(lambda fn: fn[:-9] + '_frames',
                                   one_dir_maps_files)
        maps_files.extend(one_dir_maps_files)
        info_files.extend(one_dir_info_files)
        frames_paths.extend(one_dir_frames_paths)
    
    graph = tf.Graph()
    with graph.as_default():
        dataset = tf.data.Dataset.from_tensor_slices(
            (maps_files, info_files, frames_paths))

        dataset = dataset.flat_map(lambda mf, pf, fp: 
            tf.data.Dataset.from_tensor_slices(
                tuple(tf.py_func(read_maps_poses_images, [mf, pf, fp], 
                                [tf.float32, tf.float32, tf.float32]))))
        
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        while True:
            try:
                value = sess.run(next_element)
                shape = value[0].shape
                assert shape[0] == 120
                assert shape[1] == 160
                assert shape[2] == 22
            except tf.errors.OutOfRangeError:
                break

            
            