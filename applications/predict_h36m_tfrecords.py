#!/usr/bin/python3
# -*- coding: utf-8 -*-

import __init__

import os
import tensorflow as tf
import numpy as np
import scipy
import scipy.io

import tf_pose
from tf_pose.estimator import TfPoseEstimator as OpPoseEstimator
from tf_pose.networks import get_graph_path
from pose_3d import utils

tf.logging.set_verbosity(tf.logging.WARN)


H36M_TFRECORD_PATH = '/mnt/Data/ben/tf_records_human36m/tf_records_human36m_wjoints/train'
H36M_TFRECORD_PATH_OUT = '/mnt/Data/ben/tf_records_human36m/tf_records_human36m_wjoints/train_processed'


def parse_record(record):
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

    joints2d = tf.stack([f['image/x'], f['image/y']], axis=1)
    img_raw = f['image/encoded']
    img = tf.image.decode_jpeg(img_raw)
    return img, joints2d


if __name__ == '__main__':
    tfrecord_files = sorted(os.listdir(H36M_TFRECORD_PATH))
    tfrecord_files_in = list(map(lambda f: os.path.join(H36M_TFRECORD_PATH, f),
                                 tfrecord_files))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=no-member
    estimator = OpPoseEstimator(get_graph_path('cmu'),
                                target_size=(300*2, 290*2), tf_config=config)
    for filename in tfrecord_files:
        in_file = os.path.join(H36M_TFRECORD_PATH, filename)
        record_iterator = tf.python_io.tf_record_iterator(in_file)

        heat_mats, visibilities, mask, diffs, humans = [], [], [], [], []

        for sr in record_iterator:  # one sr is one example in byte string
            img, joints2d = parse_record(sr)
            img, joints2d = tf.Session().run((img, joints2d))
            if img.shape[0] > 290:
                img = img[:290, :, :]
            if img.shape[1] > 300:
                img = img[:, :300, :]
            if img.shape[0] < 290:
                pad_0 = 290 - img.shape[0]
                img = np.pad(img, [(0, pad_0), (0, 0), (0, 0)],
                             'constant', constant_values=0)
            if img.shape[1] < 300:
                pad_1 = 300 - img.shape[1]
                img = np.pad(img, [(0, 0), (0, pad_1), (0, 0)],
                             'constant', constant_values=0)

            h = estimator.inference(img, resize_to_default=True,
                                    upsample_size=4.0)
            if len(h) == 0:
                mask.append(False)
                human = np.zeros([14, 2], dtype=np.float32)
                visibility = np.zeros([14], dtype=np.bool)
            else:
                if len(h) > 1:
                    mask.append(False)
                else:
                    mask.append(True)
                human, visibility = tf_pose.common.MPIIPart.from_coco(h[0])
                human = np.array(human)
                visibility = np.array(visibility)

            avg_location = np.mean(human[visibility], axis=0)
            avg_joint2d = np.mean(joints2d, axis=0)
            diff = np.linalg.norm(avg_location - avg_joint2d)
            diffs.append(diff)
            humans.append(human)
            visibilities.append(visibility)
            heat_mats.append(estimator.heatMat[:, :, :18])

        # Stacking on last axis makes .mat file smaller compared to first axis
        out_dict = {}
        out_dict['mask'] = np.array(mask)
        out_dict['diffs'] = np.array(diffs)
        out_dict['detected_2D'] = np.stack(humans, axis=-1)
        out_dict['visibility_2D'] = np.stack(visibilities, axis=-1)
        out_dict['heat_mat'] = np.stack(heat_mats, axis=-1)

        out_mat_filename = os.path.join(H36M_TFRECORD_PATH_OUT,
                                        filename[:-len('.tfrecord')] + '_maps')
        print(out_mat_filename)
        scipy.io.savemat(out_mat_filename, out_dict,
                         do_compression=True, appendmat=True)
