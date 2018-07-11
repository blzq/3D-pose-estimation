#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def build_model(inputs, training: bool):
    with tf.variable_scope('init_conv'):
        init_conv1 = tf.layers.conv2d(inputs, 72, [5, 5], 1)
        conv_elu1 = tf.nn.elu(init_conv1)
        init_conv2 = tf.layers.conv2d(conv_elu1, 144, [3, 3], 1)
        conv_elu2 = tf.nn.elu(init_conv2)

    with tf.variable_scope('xception1'):
        xc1 = _xception_block(conv_elu2)
    
    with tf.variable_scope('xception2'):
        xc2 = _xception_block(xc1)

    with tf.variable_scope('init_dense'):
        flatten = tf.layers.flatten(xc2)
        init_linear1 = tf.layers.dense(flatten, 72)
        linear_elu1 = tf.nn.elu(init_linear1)
        init_linear2 = tf.layers.dense(linear_elu1, 1024)
        linear_elu2 = tf.nn.elu(init_linear2)

    with tf.variable_scope('bl_res1'):
        bl1 = _bilinear_residual_block(linear_elu2, training)

    with tf.variable_scope('bl_res2'):
        bl2 = _bilinear_residual_block(bl1, training)

    with tf.variable_scope('out_fc'):
        out = tf.layers.dense(bl2, 72)

    tf.summary.histogram('out', out)

    return out


def _bilinear_residual_block(inputs, training: bool):
    linear1 = tf.layers.dense(inputs, 1024)
    bn1 = tf.layers.batch_normalization(linear1, training=training)
    elu1 = tf.nn.elu(bn1)
    dropout1 = tf.layers.dropout(elu1, training=training)

    linear2 = tf.layers.dense(dropout1, 1024)
    add = tf.add(inputs, linear2)
    bn2 = tf.layers.batch_normalization(add, training=training)
    elu2 = tf.nn.elu(bn2)
    dropout2 = tf.layers.dropout(elu2, training=training)

    return dropout2


def _xception_block(inputs):
    s_conv1 = tf.layers.separable_conv2d(inputs, 144, [3, 3], padding='same')
    relu1 = tf.nn.relu(s_conv1)

    s_conv2 = tf.layers.separable_conv2d(relu1, 144, [3, 3], padding='same')
    maxpool2 = tf.layers.max_pooling2d(s_conv2, [3, 3], 2, padding='same')

    s_conv_skip = tf.layers.conv2d(inputs, 144, [1, 1], 2)
    
    add = tf.add(s_conv_skip, maxpool2)
    return add
