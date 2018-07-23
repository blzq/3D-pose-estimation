#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import scipy.io
import cv2
import numpy as np
import tensorflow as tf

from tf_pose.common import CocoPart
from . import config


def dataset_from_filenames(maps_files, info_files, frames_paths):
    dataset = tf.data.Dataset.from_tensor_slices(
            (maps_files, info_files, frames_paths))

    dataset = dataset.interleave(lambda mf, pf, fp: 
        tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(
                read_maps_poses_images, [mf, pf, fp],
                [tf.float32, tf.float32, tf.float32, tf.float32]))),
                cycle_length=10, block_length=10)

    return dataset


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
    frames = [ cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX)
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
    locations = heatmaps_to_locations(concat)

    return concat, locations, poses, shapes


def heatmaps_to_locations(heatmaps_image_stack):
    heatmaps = heatmaps_image_stack[:, :, :, :config.n_joints]
    # heatmaps: (batch, h, w, c)
    hs = heatmaps.shape
    heatmaps_flat = np.reshape(heatmaps, [hs[0], hs[1] * hs[2], hs[3]])
    # heatmaps_flat: (batch, h * w, c)
    heatmaps_flat = np.transpose(heatmaps_flat, [0, 2, 1])
    # heatmaps_flat: (batch, c, h * w)
    argmax = np.argmax(heatmaps_flat, axis=2)
    # argmax: (batch, c)
    argmax = np.reshape(argmax, [-1])
    # argmax: (batch * c)
    locations = np.unravel_index(argmax, [hs[1], hs[2]])
    # locations: (2, batch * c)
    locations = np.transpose(locations, [1, 0])
    # locations: (batch * c, 2)
    locations = np.reshape(locations, [hs[0], hs[3], 2])
    # locations: (batch, c, 2 = [y, x])
    locations = locations.astype(np.float32)

    # Maybe don't want to do this part because information for camera is lost
    # Normalize centre of person as middle of left and right hips
    # rhip_idx = CocoPart.RHip.value
    # lhip_idx = CocoPart.LHip.value
    # centres = (locations[:, rhip_idx, :] + locations[:, lhip_idx, :]) / 2
    # locations = locations - centres[:, np.newaxis]
    # locations = np.reshape(locations, [hs[0], -1])
    # # Normalize joint locations to [-1, 1] in x and y
    # maxs = np.amax(np.abs(locations), axis=1, keepdims=True)
    # locations = locations / maxs

    return locations
