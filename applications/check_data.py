#!/usr/bin/python3
# -*- coding: utf-8 -*-

import __init__

import sys
import os
import glob

import numpy as np
import scipy.io


DATASET_PATH = '/mnt/Data/ben/surreal/SURREAL/data/cmu/train/run0/'
SUMMARY_DIR = '/home/ben/tensorflow_logs/3d_pose'
SAVER_PATH = '/home/ben/tensorflow_logs/3d_pose/ckpts/3d_pose.ckpt'

START_NAME = '01_01'  # Modify with name to start at
CHECK_HEATMAPS = True
CHECK_JOINTS = True


def read_maps(maps_file):
    print(maps_file, file=sys.stderr)
    maps_dict = scipy.io.loadmat(maps_file)
    heatmaps = np.transpose(maps_dict['heat_mat'], (3, 0, 1, 2))
        # to shape: time, height, width, n_joints = 19

    return heatmaps

def read_joints(info_file):
    print(info_file, file=sys.stderr)
    info_dict = scipy.io.loadmat(info_file)
    # in mat file - pose: [72xT], shape: [10xT], joints2D: [2x24xT]
    # reshape to T as axis 0
    joints2d = np.transpose(info_dict['joints2D'], (2, 1, 0))
    pose = info_dict['pose']

    return joints2d.astype(np.float32), pose


if __name__ == '__main__':
    dataset_dir = os.path.realpath(DATASET_PATH)
    basenames = sorted(os.listdir(dataset_dir))

    maps_files = []
    info_files = []
    frames_paths = []
    idx = basenames.index(START_NAME)
    for basename in basenames[idx:]:
        one_data_dir = os.path.join(dataset_dir, basename)
        one_dir_maps_files = glob.glob(
            os.path.join(one_data_dir, basename + '_c*_maps.mat'))
        # only get the info file and frames for heatmaps that exist
        one_dir_info_files = map(lambda f: f[:-len('_maps.mat')] + '_info.mat',
                                 one_dir_maps_files)
        one_dir_frames_paths = map(lambda f: f[:-len('_maps.mat')] + '_frames',
                                   one_dir_maps_files)
        maps_files.extend(one_dir_maps_files)
        info_files.extend(one_dir_info_files)
        frames_paths.extend(one_dir_frames_paths)

    dataset = map(read_joints, info_files)
    for maps_file, info_file in zip(maps_files, info_files):
        try:
            if CHECK_HEATMAPS:
                value = read_maps(maps_file)
                shape = value[0].shape
                assert shape[1] == 240
            if CHECK_JOINTS:
                value = read_joints(info_file)
                shape = value[0].shape
                assert len(shape) == 3
                assert shape[1] == 24
                assert shape[2] == 2
                assert (np.amax(value[1]) < 2 * np.pi)
                assert (np.amin(value[1]) > -2 * np.pi)
        except AssertionError:
            print(maps_file)
            continue
