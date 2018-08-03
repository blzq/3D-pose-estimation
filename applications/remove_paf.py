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

START_NAME = '137_14'  # Modify with name to start at


def remove_paf_from_maps(maps_file):
    print(maps_file, file=sys.stderr)
    maps_dict = scipy.io.loadmat(maps_file)
    assert 'heat_mat' in maps_dict
    assert 'detected_2D' in maps_dict
    assert 'visibility_2D' in maps_dict
    try:
        del maps_dict['paf_mat']
    except KeyError:
        return
    assert 'heat_mat' in maps_dict
    assert 'detected_2D' in maps_dict
    assert 'visibility_2D' in maps_dict
    scipy.io.savemat(os.path.join(maps_file), maps_dict,
                     do_compression=True, appendmat=True)


if __name__ == '__main__':
    dataset_dir = os.path.realpath(DATASET_PATH)
    basenames = sorted(os.listdir(dataset_dir))

    maps_files = []
    idx = basenames.index(START_NAME)
    for basename in basenames[idx:]:
        one_data_dir = os.path.join(dataset_dir, basename)
        one_dir_maps_files = glob.glob(
            os.path.join(one_data_dir, basename + '_c*_maps.mat'))
        maps_files.extend(sorted(one_dir_maps_files))

    for maps_file in maps_files:
        remove_paf_from_maps(maps_file)
