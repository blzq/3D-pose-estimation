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
    pm_3d = PoseModel3d(os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                        'logs'))
    basename = 'ung_143_26'
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                            basename)
    maps_files = glob.glob(os.path.join(data_dir, 
                                        basename + '_c*_maps.mat'))
    info_files = glob.glob(os.path.join(data_dir, 
                                        basename + '_c*_info.mat'))
    frames_paths = glob.glob(os.path.join(data_dir, 
                                        basename + '_c*_frames'))
    dataset = tf.data.Dataset.from_tensor_slices(
        (maps_files, info_files, frames_paths))

    dataset = dataset.flat_map(lambda mf, pf, fp: 
        tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(read_maps_poses_images, [mf, pf, fp], 
                             [tf.float32, tf.float32, tf.float32]))))
    with pm_3d.graph.as_default():
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        print(pm_3d.sess.run(next_element))

    # in_array = np.array([1, 2, 3])[np.newaxis]
    # pm_3d.train(in_array, np.array([2, 4, 6])[np.newaxis])
    # print(pm_3d.estimate(in_array))