#!/usr/bin/python3
# -*- coding: utf-8 -*-
import __init__

import sys
import os

from tf_pose.estimator import TfPoseEstimator as OpPoseEstimator
from tf_pose.networks import get_graph_path
import tf_pose.common

import numpy as np
import scipy.io
import cv2

import matplotlib.pyplot as plt

def main(surreal_path):
    estimator = OpPoseEstimator(get_graph_path('cmu'))
    base_path = os.path.join(surreal_path, 'data', 'cmu', 'train')
    for run in ['run0', 'run1', 'run2']:
        run_path = os.path.join(base_path, run)
        dir_names = sorted(os.listdir(run_path))
        for dir_name in dir_names:
            dir_path = os.path.join(run_path, dir_name)
            video_filenames = [ f for f in os.listdir(dir_path)
                                if f.endswith('.mp4') ]
            for video_filename in video_filenames:
                video_in = os.path.join(dir_path, video_filename)
                print(video_in)
                process_video(video_in, dir_path, estimator)
    return


def process_video(in_file, out_dir, estimator):
    capture = cv2.VideoCapture(in_file)

    humans = []
    visibilities = []
    heat_mats = []
    paf_mats = []
    n = 0
    while True:
        n += 1
        retval = capture.grab()
        if not retval:
            break
        _, color_im = capture.retrieve()
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)

        human = estimator.inference(color_im, upsample_size=4.0)

        try:
            human = human[0]
        except IndexError:
            with open(os.path.join(
                        os.path.realpath(os.path.dirname(__file__)),
                        'invalid.txt'),
                      'a') as f:
                f.write(in_file + '\n')
                return

        human, visibility = tf_pose.common.MPIIPart.from_coco(human)

        humans.append(np.array(human))
        visibilities.append(np.array(visibility))
        heat_mats.append(estimator.heatMat)
        paf_mats.append(estimator.pafMat)

    out_dict = {}
    out_dict['detected_2D'] = np.stack(humans, axis=-1)
    out_dict['visibility_2D'] = np.stack(visibilities, axis=-1)
    out_dict['heat_mat'] = np.stack(heat_mats, axis=-1)
    out_dict['paf_mat'] = np.stack(paf_mats, axis=-1)

    out_mat_filename = \
        os.path.splitext(os.path.basename(in_file))[0] + '_maps'
    scipy.io.savemat(os.path.join(out_dir, out_mat_filename), out_dict,
                        do_compression=True, appendmat=True)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 predict_surreal_videos.py <path-to-SURREAL-dataset>")
    surreal_path = os.path.realpath(sys.argv[1])
    sys.exit(main(surreal_path))
