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


if __name__ == '__main__':
    basenames = ['ung_143_26']  # Use os.listdir() in future
    dataset_dir = os.path.dirname(os.path.realpath(__file__))  # path to all data folders
    maps_files = []
    info_files = []
    frames_paths = []
    for basename in basenames:
        data_dir = os.path.join(dataset_dir, basename)
        maps_files.append(glob.glob(os.path.join(data_dir,
                                                 basename + '_c*_maps.mat')))
        info_files.append(glob.glob(os.path.join(data_dir, 
                                                 basename + '_c*_info.mat')))
        frames_paths.append(glob.glob(os.path.join(data_dir,
                                                   basename + '_c*_frames')))

    pm_3d = PoseModel3d((None, 120, 160, 19),
                        72)

    with pm_3d.graph.as_default():
        dataset = tf.data.Dataset.from_tensor_slices(
            (maps_files, info_files, frames_paths))

        dataset = dataset.flat_map(lambda mf, pf, fp: 
            tf.data.Dataset.from_tensor_slices(
                tuple(tf.py_func(read_maps_poses_images, [mf, pf, fp], 
                                [tf.float32, tf.float32, tf.float32]))))
    pm_3d.train(dataset, epochs=5, batch_size=2)

    # in_array = np.array([1, 2, 3])[np.newaxis]
    # pm_3d.train(in_array, np.array([2, 4, 6])[np.newaxis])
    # print(pm_3d.estimate(in_array))