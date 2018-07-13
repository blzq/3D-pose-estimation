#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import scipy.io
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

    frames = [ cv2.cvtColor(cv2.imread(f.decode('utf-8')), cv2.COLOR_BGR2RGB)
               for f in glob.glob(frames_path + b'/f*.jpg') ]
    frames = [ cv2.normalize(frame.astype('float'), None, 
                             0.0, 1.0, cv2.NORM_MINMAX)
               for frame in frames ]
    frames = [ cv2.resize(frame, 
                          dsize=(heatmaps.shape[2], heatmaps.shape[1]),
                          interpolation=cv2.INTER_AREA)
               for frame in frames ]
    frames = np.array(frames, dtype=np.float32)
        # shape: time, 240, 320 halved in x, y dims

    min_length = np.min([frames.shape[0], poses.shape[0], heatmaps.shape[0]])
    heatmaps = heatmaps[:min_length]
    poses = poses[:min_length]
    shapes = shapes[:min_length]
    frames = frames[:min_length]

    concat = np.concatenate([heatmaps, frames], axis=3)

    return concat, poses, shapes

