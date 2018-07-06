#!/usr/bin/python3
# -*- coding: utf-8 -*-

import __init__

import sys
import os
import glob

import tensorflow as tf
import numpy as np

from pose_3d.pose_model_3d import PoseModel3d
from pose_3d.data_helpers import *

DATA_PATH = '/mnt/Data/ben/surreal/SURREAL/data/cmu/train/run0/'

if __name__ == '__main__':
    dataset_dir = os.path.realpath(DATA_PATH)
    basenames = os.listdir(dataset_dir)

    maps_files = []
    info_files = []
    frames_paths = []
    for basename in basenames[:1]:
        one_data_dir = os.path.join(dataset_dir, basename)
        one_dir_maps_files = glob.glob(os.path.join(one_data_dir,
                                                 basename + '_c*_maps.mat'))
        one_dir_info_files = map(lambda fn: fn[:-9] + '_info.mat', 
                                 one_dir_maps_files)
        one_dir_frames_paths = map(lambda fn: fn[:-9] + '_frames',
                                   one_dir_maps_files)
        maps_files.extend(one_dir_maps_files)
        info_files.extend(one_dir_info_files)
        frames_paths.extend(one_dir_frames_paths)

    pm_3d = PoseModel3d((None, 120, 160, 19),
                        72,
                        summary_dir='~/tensorflow_logs/3d_pose',
                        saver_path='~/tensorflow_ckpts/3d_pose',
                        restore_model=False)

    with pm_3d.graph.as_default():
        dataset = tf.data.Dataset.from_tensor_slices(
            (maps_files, info_files, frames_paths))

        dataset = dataset.flat_map(lambda mf, pf, fp: 
            tf.data.Dataset.from_tensor_slices(
                tuple(tf.py_func(read_maps_poses_images, [mf, pf, fp], 
                                [tf.float32, tf.float32, tf.float32]))))
    pm_3d.train(dataset, epochs=5, batch_size=16)

    # in_array = np.array([1, 2, 3])[np.newaxis]
    # pm_3d.train(in_array, np.array([2, 4, 6])[np.newaxis])
    # print(pm_3d.estimate(in_array))