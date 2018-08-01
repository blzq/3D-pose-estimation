import os
import glob
import scipy.io
import cv2
import numpy as np
import tensorflow as tf

from tf_pose import common
from . import config


def dataset_from_filenames(maps_files, info_files, frames_paths):
    dataset = tf.data.Dataset.from_tensor_slices(
            (maps_files, info_files, frames_paths))

    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        lambda mf, pf, fp: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(read_maps_poses_images, [mf, pf, fp],
                             [tf.float32, tf.float32, tf.float32,
                              tf.float32, tf.float32]))),
        cycle_length=6, block_length=1, sloppy=True,
        buffer_output_elements=32, prefetch_input_elements=6))

    return dataset


def read_maps_poses_images(maps_file, info_file, frames_path):
    maps_dict = scipy.io.loadmat(maps_file)
    heatmaps = np.transpose(maps_dict['heat_mat'], (3, 0, 1, 2))
        # to shape: time, height, width, n_joints = 19
    # Flip heatmap horizontally because image and 3D GT are flipped in SURREAL
    heatmaps = np.flip(heatmaps, axis=2)
    img_size_x = heatmaps.shape[2]

    info_dict = scipy.io.loadmat(info_file)
    # in mat file - pose: [72xT], shape: [10xT], joints2D: [2x24xT]
    # reshape to T as axis 0
    poses = np.transpose(info_dict['pose'], (1, 0))
    shapes = np.transpose(info_dict['shape'], (1, 0))
    joints2d = np.transpose(info_dict['joints2D'], (2, 1, 0))
    # Flip 2D GT horizontally because image and 3D GT are flipped in SURREAL
    joints2d = img_size_x - joints2d

    frames = [ cv2.cvtColor(cv2.imread(f.decode('utf-8')), cv2.COLOR_BGR2RGB)
               for f in glob.glob(frames_path + b'/f*.jpg') ]
    frames = [ cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX)
               for frame in frames ]
    # Flip image horizontally because image and 3D GT are flipped in SURREAL
    frames = np.flip(frames, axis=2)
    # frames = [ cv2.resize(frame,
    #                       dsize=(heatmaps.shape[2], heatmaps.shape[1]),
    #                       interpolation=cv2.INTER_AREA)
    #            for frame in frames ]
    frames = np.array(frames, dtype=np.float32)
        # shape: time, 240, 320

    min_length = np.min([frames.shape[0], poses.shape[0], heatmaps.shape[0]])
    heatmaps = heatmaps[:min_length]
    poses = poses[:min_length]
    shapes = shapes[:min_length]
    joints2d = joints2d[:min_length]
    frames = frames[:min_length]

    concat = np.concatenate([heatmaps, frames], axis=3)
    locations = heatmaps_to_locations(concat)

    return concat, locations, poses, shapes, joints2d.astype(np.float32)


def heatmaps_to_locations(heatmaps_image_stack):
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
    # Move centre of image to (0, 0)
    img_dim = hs[1:3]
    np.subtract(locations, img_dim, out=locations)
    # Scale detection locations by shorter side length
    img_side_length = np.amin(img_dim)
    np.divide(locations, img_side_length, out=locations)

    locations_with_vals = np.concatenate([locations, max_val], axis=2)

    # Maybe don't want to do this part because information for camera is lost
    # Normalize centre of person as middle of left and right hips
    # rhip_idx = common.CocoPart.RHip.value
    # lhip_idx = common.CocoPart.LHip.value
    # centres = (locations[:, rhip_idx, :] + locations[:, lhip_idx, :]) / 2
    # locations = locations - centres[:, np.newaxis]
    # locations = np.reshape(locations, [hs[0], -1])
    # # Normalize joint locations to [-1, 1] in x and y
    # maxs = np.amax(np.abs(locations), axis=1, keepdims=True)
    # locations = locations / maxs

    return locations_with_vals

def suppress_non_largest_human(humans, heatmaps, expected_in_size):
    d = 8  # padding around other humans to suppress
    largest_human_size = 0
    human_extents = []
    largest_human_idx = -1
    # Find human extents
    for h_idx, human in enumerate(humans):
        min_x, max_x = float('inf'), 0
        min_y, max_y = float('inf'), 0
        for i in range(common.CocoPart.Background.value):
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
