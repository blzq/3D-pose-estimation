#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import scipy.io
import imageio
import cv2
import numpy as np
import tensorflow as tf

def read_maps_poses_images(maps_file, info_file, frames_path):
    maps_dict = scipy.io.loadmat(maps_file)
    heatmaps = np.transpose(maps_dict['heat_mat'], (3, 0, 1, 2))
        # to shape: time, height, width, n_joints = 19

    info_dict = scipy.io.loadmat(info_file)
    poses = np.transpose(info_dict['pose'], (1, 0))
        # to shape: time, 72 = 24 joints x 3 axis-angle rot
    shapes = np.transpose(info_dict['shape'], (1, 0))

    frames = [ cv2.imread(str(f))
               for f in glob.glob(frames_path + b'/f*.jpg') ]
    frames = np.array(frames, dtype=np.float32)
        # shape: time, 240, 320

    min_length = np.min([frames.shape[0], poses.shape[0], heatmaps.shape[0]])
    heatmaps = heatmaps[:min_length]
    poses = poses[:min_length]
    frames = frames[:min_length]

    return heatmaps, poses, shapes, frames

