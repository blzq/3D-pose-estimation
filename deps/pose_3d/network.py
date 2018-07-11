#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def build_model(inputs, training: bool):
    with tf.variable_scope('init'):
        inputs = inputs[:, :18]
        init_conv1 = tf.layers.conv2d(inputs, 144, [5, 5], 1)
        conv_elu1 = tf.nn.elu(init_conv1)
        init_conv2 = tf.layers.conv2d(conv_elu1, 72, [3, 3], 1)
        init_pool = tf.layers.max_pooling2d(init_conv2, 3, 1)
        conv_elu2 = tf.nn.elu(init_pool)
        init_bn = tf.layers.batch_normalization(conv_elu2, training=training)
        init_drop = tf.layers.dropout(init_bn, rate=0.2, training=training)
        flatten = tf.layers.flatten(init_drop)
        init_linear1 = tf.layers.dense(flatten, 72)
        linear_elu1 = tf.nn.elu(init_linear1)
        init_linear2 = tf.layers.dense(linear_elu1, 1024)
        linear_elu2 = tf.nn.elu(init_linear2)

    with tf.variable_scope('bl_res1'):
        bl1 = _bilinear_residual_block(linear_elu2, training)

    with tf.variable_scope('bl_res2'):
        bl2 = _bilinear_residual_block(bl1, training)

    with tf.variable_scope('out_fc'):
        out = tf.layers.dense(bl2, 72,
                              kernel_initializer=tf.initializers.zeros)

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