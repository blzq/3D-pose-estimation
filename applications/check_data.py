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

START_NAME = 'ung_86_02'  # Modify with name to start at


def read_maps(maps_file):
    print(maps_file, file=sys.stderr)
    maps_dict = scipy.io.loadmat(maps_file)
    heatmaps = np.transpose(maps_dict['heat_mat'], (3, 0, 1, 2))
        # to shape: time, height, width, n_joints = 19

    return heatmaps, maps_file


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
        maps_files.extend(one_dir_maps_files)

    dataset = map(read_maps, maps_files)
    for value in dataset:
        try:
            shape = value[0].shape
            filename = value[1]
            assert shape[1] == 240
        except AssertionError:
            print(filename)
            continue
