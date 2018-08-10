#!/usr/bin/python3
# -*- coding: utf-8 -*-
import __init__

import sys
import os
import glob

from tf_pose.estimator import TfPoseEstimator as OpPoseEstimator
from tf_pose.networks import get_graph_path
import tf_pose.common
from pose_3d.data_helpers import heatmaps_to_locations

import numpy as np
import scipy.io
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


def main(surreal_path):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    estimator = OpPoseEstimator(get_graph_path('cmu'),
                                target_size=(320*2, 240*2), tf_config=config)
    base_path = os.path.join(surreal_path, 'data', 'cmu', 'train')
    for run in ['run0']:  # + ['run1', 'run2']:
        run_path = os.path.join(base_path, run)
        dir_names = sorted(os.listdir(run_path))
        for dir_name in dir_names:
            dir_path = os.path.join(run_path, dir_name)
            one_dir_frames_path = sorted(glob.glob(
                os.path.join(dir_path, dir_name + '*_frames')))
            for frames_path in one_dir_frames_path:
                process_frames(frames_path, estimator)


def process_frames(in_path, estimator):
    print(in_path)
    basename = in_path[:-len("_frames")]
    info_filename = basename + '_info.mat'
    info_dict = scipy.io.loadmat(info_filename)
    joints2d = np.transpose(info_dict['joints2D'], (2, 1, 0)) # T, 24, 2
    frames_files = [ os.path.join(in_path, f) 
                     for f in sorted(os.listdir(in_path)) ]
    assert len(frames_files) == joints2d.shape[0]

    humans = []
    visibilities = []
    heat_mats = []
    diffs = []
    mask = []
    for idx, frame_file in enumerate(frames_files):
        color_im = cv2.imread(frame_file)
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
        human = estimator.inference(color_im,
                                    resize_to_default=True, upsample_size=4.0)
        if len(human) == 0:
            mask.append(False)
            human = np.zeros([14, 2], dtype=np.float32)
            visibility = np.zeros([14], dtype=np.bool)
        else:
            if len(human) > 1:
                mask.append(False)
            else:
                mask.append(True)
            human, visibility = tf_pose.common.MPIIPart.from_coco(human[0])
            human = np.array(human)
            visibility = np.array(visibility)
        avg_location = np.mean(human[visibility], axis=0)
        avg_joint2d = np.mean(joints2d[idx], axis=0)
        diff = np.linalg.norm(avg_location - avg_joint2d)
        diffs.append(diff)
        humans.append(human)
        visibilities.append(visibility)
        heat_mats.append(estimator.heatMat)

    out_dict = {}
    out_dict['mask'] = np.array(mask)
    out_dict['diffs'] = np.array(diffs)
    out_dict['detected_2D'] = np.stack(humans, axis=-1)
    out_dict['visibility_2D'] = np.stack(visibilities, axis=-1)
    out_dict['heat_mat'] = np.stack(heat_mats, axis=-1)

    out_mat_filename = basename + '_maps'
    print(out_mat_filename)
    scipy.io.savemat(out_mat_filename, out_dict, 
                     do_compression=True, appendmat=True)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 predict_surreal_videos.py <path-to-SURREAL-dataset>")
        sys.exit()
    surreal_path = os.path.realpath(sys.argv[1])
    sys.exit(main(surreal_path))
