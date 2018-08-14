#!/usr/bin/python3
# -*- coding: utf-8 -*-

import __init__

import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


H36M_TFRECORD_PATH = '/mnt/Data/ben/tf_records_human36m/tf_records_human36m_wjoints/train'


def parse_record(record_file):
    dict_keys = {'image/center': tf.FixedLenFeature([2], tf.int64),
                 'image/encoded': tf.VarLenFeature(tf.string),
                 'image/filename': tf.VarLenFeature(tf.string),
                 'image/format': tf.VarLenFeature(tf.string),
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

    f = tf.parse_single_example(record_file, dict_keys)

    joints2d = tf.stack(f['image/x'], f['image/y'], axis=1)
    image = tf.image.decode_image(f['image/encoded'])
    return image, f['mosh/pose'], f['mosh/shape'], joints2d


if __name__ == '__main__':
    tfrecord_files = os.listdir(H36M_TFRECORD_PATH)
    tfrecord_files = tfrecord_files[:1]
    tfrecord_files = list(map(lambda f: os.path.join(H36M_TFRECORD_PATH, f),
                              tfrecord_files))

    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_record)
    iterator = dataset.make_initializable_iterator()
    sess = tf.Session()
    sess.run(iterator.initializer, feed_dict={filenames: tfrecord_files})
    next_record = iterator.get_next()
    record = sess.run(next_record)
    print(record)
