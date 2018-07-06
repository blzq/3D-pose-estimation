#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def build_model(inputs, training: bool):
    with tf.variable_scope('init_conv'):
        init_conv1 = tf.layers.separable_conv2d(inputs, 72, [3, 3], 2)
        conv_relu1 = tf.nn.relu(init_conv1)
        init_conv2 = tf.layers.separable_conv2d(conv_relu1, 72, [3, 3], 2)
        conv_relu2 = tf.nn.relu(init_conv2)
        flatten = tf.layers.flatten(conv_relu2)
        init_linear = tf.layers.dense(flatten, 1024)
        linear_relu = tf.nn.relu(init_linear)

    with tf.variable_scope('bl_res1'):
        bl1 = _bilinear_residual_block(linear_relu, training)

    with tf.variable_scope('bl_res2'):
        bl2 = _bilinear_residual_block(bl1, training)

    with tf.variable_scope('out_fc'):
        out = tf.layers.dense(bl2, 72)
        
    return out
    

def _bilinear_residual_block(inputs, training: bool):
    linear1 = tf.layers.dense(inputs, 1024)
    bn1 = tf.layers.batch_normalization(linear1, training=training)
    relu1 = tf.nn.relu(bn1)
    dropout1 = tf.layers.dropout(relu1, training=training)

    linear2 = tf.layers.dense(dropout1, 1024)
    bn2 = tf.layers.batch_normalization(linear2, training=training)
    relu2 = tf.nn.relu(bn2)
    dropout2 = tf.layers.dropout(relu2, training=training)

    add = tf.add(inputs, dropout2)
    return add