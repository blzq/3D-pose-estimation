# -*- coding: utf-8 -*-

import glob
import scipy.io
import cv2
import numpy as np
import tensorflow as tf

import tf_pose.common
from . import config


def dataset_from_filenames_surreal(maps_files, info_files, frames_paths):
    dataset = tf.data.Dataset.from_tensor_slices(
            (maps_files, info_files, frames_paths))

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            lambda mf, pf, fp: tf.data.Dataset.from_tensor_slices(tuple(
                tf.py_func(read_maps_poses_images_surreal,
                           [mf, pf, fp], [tf.float32] * 5, stateful=False))),
        cycle_length=12, block_length=1, sloppy=True,
        buffer_output_elements=32, prefetch_input_elements=4))

    return dataset


def dataset_from_filenames_h36m(maps_files, record_files):
    record_dataset = tf.data.TFRecordDataset(record_files)
    record_dataset = record_dataset.map(read_tfrecord_h36m)

    heatmaps_dataset = tf.data.Dataset.from_tensor_slices(maps_files)
    heatmaps_dataset = heatmaps_dataset.flat_map(
        lambda mf: tf.data.Dataset.from_tensor_slices(
            tf.py_func(read_maps_h36m, [mf], tf.float32, stateful=False)))
    heatmaps_dataset = heatmaps_dataset.prefetch(1)

    h36m_dataset = tf.data.Dataset.zip((heatmaps_dataset, record_dataset))
    h36m_dataset = h36m_dataset.map(concat_img_heatmaps_h36m)
    return h36m_dataset


def read_maps_poses_images_surreal(maps_file, info_file, frames_path):
    maps_dict = scipy.io.loadmat(maps_file)
    # to shape: time, height, width, n_joints
    heatmaps = np.transpose(maps_dict['heat_mat'], (3, 0, 1, 2))
    mask = np.squeeze(maps_dict['mask'])
    diffs = np.squeeze(maps_dict['diffs'])
    # diffs can contain NaNs but the '<' op should exclude them
    np.logical_and(mask, diffs < 250, out=mask)
    # Flip heatmap horizontally because image and 3D GT are flipped in SURREAL
    heatmaps = np.flip(heatmaps, axis=2)
    # Reorder heatmaps - swap lefts and rights for same reason (see config.py)
    reord = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16, 18]
    heatmaps = heatmaps[:, :, :, reord]
    heatmaps = heatmaps[:, :, :, :config.n_joints]
    img_size_x = heatmaps.shape[2]

    info_dict = scipy.io.loadmat(info_file)
    # in mat file - pose: [72xT], shape: [10xT], joints2D: [2x24xT]
    # reshape to T as axis 0
    poses = np.transpose(info_dict['pose'], (1, 0))
    shapes = np.transpose(info_dict['shape'], (1, 0))
    # to shape: time, joints, (x, y)
    joints2d = np.transpose(info_dict['joints2D'], (2, 1, 0))
    # Flip 2D GT horizontally because image and 3D GT are flipped in SURREAL
    joints2d[:, :, 0] = img_size_x - joints2d[:, :, 0]
    zrot = np.squeeze(np.array(info_dict['zrot']))

    # Make sure to sort the frames: VERY IMPORTANT!
    frames = [ cv2.cvtColor(cv2.imread(f.decode('utf-8')), cv2.COLOR_BGR2RGB)
               for f in sorted(glob.glob(frames_path + b'/f*.jpg')) ]
    frames = np.array(frames, dtype=np.float32)
    frames = [ cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX)
               for frame in frames ]
    # Flip image horizontally because image and 3D GT are flipped in SURREAL
    frames = np.flip(frames, axis=2)

    concat = np.concatenate([heatmaps, frames], axis=3)

    concat, poses, shapes, joints2d, zrot = (
        concat[mask], poses[mask], shapes[mask], joints2d[mask], zrot[mask])

    skip = 2  # Only take every n-th frame
    concat, poses, shapes, joints2d, zrot = (
        concat[::skip], poses[::skip], shapes[::skip], joints2d[::skip],
        zrot[::skip])

    return concat, poses, shapes, joints2d.astype(np.float32), zrot


def read_tfrecord_h36m(record):
    # These are the tfrecords provided by https://github.com/akanazawa/hmr
    dict_keys = {'image/center': tf.FixedLenFeature([2], tf.int64),
                 'image/encoded': tf.FixedLenFeature([], tf.string),
                 'image/filename': tf.FixedLenFeature([1], tf.string),
                 'image/format': tf.FixedLenFeature([1], tf.string),
                 'image/height': tf.FixedLenFeature([1], tf.int64),
                 'image/visibility': tf.FixedLenFeature([14], tf.int64),
                 'image/width': tf.FixedLenFeature([1], tf.int64),
                 'image/x': tf.FixedLenFeature([14], tf.float32),
                 'image/y': tf.FixedLenFeature([14], tf.float32),
                 'meta/crop_pt': tf.FixedLenFeature([2], tf.int64),
                 'meta/has_3d': tf.FixedLenFeature([1], tf.int64),
                 'meta/scale_factors': tf.FixedLenFeature([2], tf.float32),
                 'mosh/gt3d': tf.FixedLenFeature([14 * 3], tf.float32),
                 'mosh/pose': tf.FixedLenFeature([72], tf.float32),
                 'mosh/shape': tf.FixedLenFeature([10], tf.float32)}

    f = tf.parse_single_example(record, dict_keys)

    poses = f['mosh/pose']
    shapes = f['mosh/shape']
    joints2d = tf.stack([f['image/x'], f['image/y']], axis=1)
    img_raw = f['image/encoded']
    img = tf.image.decode_jpeg(img_raw)
    return img, poses, shapes, joints2d


def read_maps_h36m(maps_file):
    # These are the heatmaps generated using predict_h36m_tfrecords.py
    maps_dict = scipy.io.loadmat(maps_file)
    heatmaps = np.transpose(maps_dict['heat_mat'], (3, 0, 1, 2))
    heatmaps = heatmaps[:, :, :, :config.n_joints]
    return heatmaps


def concat_img_heatmaps_h36m(heatmaps, img, poses, shapes, joints2d):
    return tf.concat([heatmaps, img], axis=3), poses, shapes, joints2d, 0.0


def heatmaps_to_locations(heatmaps_image_stack):
    # Currently unused in favour of utils.soft_argmax_rescaled
    heatmaps = heatmaps_image_stack[:, :, :, :config.n_joints]
    # heatmaps: (batch, h, w, c)
    hs = heatmaps.shape
    heatmaps_flat = np.reshape(heatmaps, [hs[0], hs[1] * hs[2], hs[3]])
    # heatmaps_flat: (batch, h * w, c)
    heatmaps_flat = np.transpose(heatmaps_flat, [0, 2, 1])
    # heatmaps_flat: (batch, c, h * w)
    argmax = np.argmax(heatmaps_flat, axis=2)
    # https://stackoverflow.com/questions/5798364/using-numpy-argmax-on-multidimensional-arrays
    a1, a2 = np.indices(argmax.shape)
    max_val = heatmaps_flat[a1, a2, argmax]
    max_val = max_val[..., np.newaxis]
    # argmax: (batch, c); max_val: (batch, c)
    argmax = np.reshape(argmax, [-1])
    # argmax: (batch * c)
    locations = np.unravel_index(argmax, [hs[1], hs[2]])
    # locations: (2, batch * c)
    locations = np.transpose(locations, [1, 0])
    # locations: (batch * c, 2)
    locations = np.reshape(locations, [hs[0], hs[3], 2])
    # locations: (batch, c, 2 = [y, x])
    locations = locations.astype(np.float32)

    # Normalize detection locations by image size
    # Move centre of image to (0, 0) and
    # scale detection locations by shorter side length
    # img_dim = np.array(hs[1:3], dtype=np.float32)
    # img_side_length = np.minimum(img_dim[0], img_dim[1])
    # np.divide(img_dim, 2, out=img_dim)
    # np.subtract(locations, img_dim, out=locations)
    # np.divide(locations, img_side_length, out=locations)

    locations_with_vals = np.concatenate([locations, max_val], axis=2)

    # Maybe don't want to do this part because information for camera is lost
    # Normalize centre of person as middle of left and right hips
    # and normalize joint locations to [-1, 1]
    # rhip_idx = tf_pose.common.CocoPart.RHip.value
    # lhip_idx = tf_pose.common.CocoPart.LHip.value
    # centres = (locations[:, rhip_idx, :] + locations[:, lhip_idx, :]) / 2
    # locations = locations - centres[:, np.newaxis]
    # maxs = np.amax(np.abs(locations), axis=[1, 2], keepdims=True)
    # locations = locations / maxs

    return locations_with_vals


def suppress_non_largest_human(humans, heatmaps, expected_in_size):
    d = 10  # padding around other humans to also suppress
    largest_human_size = 0
    human_extents = []
    largest_human_idx = -1
    # Find human extents
    for h_idx, human in enumerate(humans):
        min_x, max_x = float('inf'), 0
        min_y, max_y = float('inf'), 0
        for i in range(tf_pose.common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue
            body_part = human.body_parts[i]
            center = (int(body_part.x * expected_in_size[1] + 0.5),
                      int(body_part.y * expected_in_size[0] + 0.5))
            max_x = max(center[0], max_x)
            min_x = min(center[0], min_x)
            max_y = max(center[1], max_y)
            min_y = min(center[1], min_y)
        human_extents.append((min_x, max_x, min_y, max_y))
        one_largest_range = (max_x - min_x) * (max_y - min_y)
        if one_largest_range > largest_human_size:
            largest_human_size = one_largest_range
            largest_human_idx = h_idx

    # Suppress humans that are not the largest human in the image
    for h_idx, extent in enumerate(human_extents):
        if h_idx != largest_human_idx:
            min_e_x, max_e_x, min_e_y, max_e_y = extent
            heatmaps[min_e_y-d : max_e_y+d, min_e_x-d : max_e_x+d] = 0
    return heatmaps
