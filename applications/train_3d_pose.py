#!/usr/bin/python3
# -*- coding: utf-8 -*-

import __init__

import sys
import os
import glob

import tensorflow as tf
import numpy as np

from pose_3d.pose_model_3d import PoseModel3d
from pose_3d.data_helpers import read_maps_poses_images

DATASET_PATH = '/mnt/Data/ben/surreal/SURREAL/data/cmu/train/run0/'
SUMMARY_DIR = '/home/ben/tensorflow_logs/3d_pose'
SAVER_PATH = '/home/ben/tensorflow_ckpts/3d_pose'

if __name__ == '__main__':
    dataset_dir = os.path.realpath(DATASET_PATH)
    basenames = os.listdir(dataset_dir)

    smpl_path = os.path.join(
        __init__.project_path, 'data', 'SMPL_model', 'models_numpy')
    smpl_neutral = os.path.join(smpl_path, 'model_neutral_np.pkl')

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

    pm_3d = PoseModel3d((None, 120, 160, 22),
                        training=True,
                        summary_dir=SUMMARY_DIR,
                        saver_path=SAVER_PATH,
                        restore_model=True,
                        mesh_loss=True,
                        smpl_model=smpl_neutral)

    with pm_3d.graph.as_default():
        dataset = tf.data.Dataset.from_tensor_slices(
            (maps_files, info_files, frames_paths))

        dataset = dataset.flat_map(lambda mf, pf, fp: 
            tf.data.Dataset.from_tensor_slices(
                tuple(tf.py_func(read_maps_poses_images, [mf, pf, fp], 
                                [tf.float32, tf.float32, tf.float32, tf.float32]))))
    pm_3d.train(dataset, epochs=10000, batch_size=20)
