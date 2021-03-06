#!/usr/bin/python3
# -*- coding: utf-8 -*-

import __init__

import os
import glob
import random

import tensorflow as tf

from pose_3d.pose_model_3d import PoseModel3d
from pose_3d.data_helpers import dataset_from_filenames_surreal
from pose_3d import config


DATASET_PATH = '/mnt/Data/ben/surreal/SURREAL/data/cmu/train/run0/'
SUMMARY_DIR = '/home/ben/tensorflow_logs/3d_pose'
SAVER_PATH = '/home/ben/tensorflow_logs/3d_pose/ckpts/3d_pose.ckpt'


if __name__ == '__main__':
    dataset_dir = os.path.realpath(DATASET_PATH)
    basenames = sorted(os.listdir(dataset_dir))

    smpl_path = os.path.join(
        __init__.project_path, 'data', 'SMPL_model', 'models_numpy')
    smpl_neutral = os.path.join(smpl_path, 'model_neutral_np.pkl')

    maps_files = []
    info_files = []
    frames_paths = []
    for basename in basenames:
        # each basename is one dir corresponding to one type of action
        one_data_dir = os.path.join(dataset_dir, basename)
        one_dir_maps_files = sorted(glob.glob(
            os.path.join(one_data_dir, basename + '_c*_maps.mat')))
        # only get the info file and frames for heatmaps that exist
        one_dir_info_files = map(lambda f: f[:-len('_maps.mat')] + '_info.mat',
                                 one_dir_maps_files)
        one_dir_frames_paths = map(lambda f: f[:-len('_maps.mat')] + '_frames',
                                   one_dir_maps_files)
        maps_files.extend(one_dir_maps_files)
        info_files.extend(one_dir_info_files)
        frames_paths.extend(one_dir_frames_paths)

    # Shuffle file order
    all_files = list(zip(maps_files, info_files, frames_paths))
    random.shuffle(all_files)
    maps_files, info_files, frames_paths = zip(*all_files)
    maps_files, info_files, frames_paths = (
        list(maps_files), list(info_files), list(frames_paths))

    graph = tf.Graph()
    with graph.as_default():
        dataset = dataset_from_filenames_surreal(
            maps_files, info_files, frames_paths)

    pm_3d = PoseModel3d((None, 240, 320, 3 + config.n_joints),
                        graph,
                        mode='train',
                        dataset=dataset,
                        summary_dir=SUMMARY_DIR,
                        saver_path=SAVER_PATH,
                        restore_model=True,
                        pose_loss=True,
                        mesh_loss=True,
                        reproject_loss=True,
                        smpl_model=smpl_neutral,
                        discriminator=False)

    pm_3d.train(batch_size=32, epochs=500)
