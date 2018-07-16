#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_rodrigues.rodrigues import rodrigues_batch
from tf_pose.common import CocoPart


def build_model(inputs, inputs_locs, training: bool):
    with tf.variable_scope('encoder'):
        with tf.variable_scope('init_conv'):
            init_conv1 = tf.layers.conv2d(inputs, 36, [3, 3], 1)
            bn1 = tf.layers.batch_normalization(init_conv1, training=training)
            conv_relu1 = tf.nn.relu(bn1)
            init_conv2 = tf.layers.conv2d(conv_relu1, 72, [3, 3], 1)
            bn2 = tf.layers.batch_normalization(init_conv2, training=training)
            conv_relu2 = tf.nn.relu(bn2)

        with tf.variable_scope('xception1'):
            xc1 = _xception_block(conv_relu2, training)
        
        with tf.variable_scope('xception2'):
            xc2 = _xception_block(xc1, training)

        with tf.variable_scope('xception3'):
            xc3 = _xception_block(xc2, training)
        
        with tf.variable_scope('xception_out'):
            xc_out = tf.layers.separable_conv2d(xc3, 144, [3, 3])
            xc_relu = tf.nn.relu(xc_out)
            xc_pool = tf.reduce_mean(xc_relu, axis=[1, 2])

        with tf.variable_scope('init_dense'):
            features_flat = tf.layers.flatten(xc_pool)
            locations_flat = tf.layers.flatten(inputs_locs)
            in_concat = tf.concat([features_flat, locations_flat], axis=1)
            init_linear1 = tf.layers.dense(in_concat, 144)
            linear_relu1 = tf.nn.relu(init_linear1)
            init_linear2 = tf.layers.dense(linear_relu1, 1024)
            linear_relu2 = tf.nn.relu(init_linear2)

        with tf.variable_scope('bl_res1'):
            bl1 = _bilinear_residual_block(linear_relu2, training)

        with tf.variable_scope('bl_res2'):
            bl2 = _bilinear_residual_block(bl1, training)

        with tf.variable_scope('bl_res3'):
            bl3 = _bilinear_residual_block(bl2, training)

        with tf.variable_scope('out_fc'):
            out = tf.layers.dense(bl3, 72)

        tf.summary.histogram('out', out)

    return out


def _bilinear_residual_block(inputs, training: bool):
    linear1 = tf.layers.dense(inputs, 1024)
    bn1 = tf.layers.batch_normalization(linear1, training=training)
    relu1 = tf.nn.relu(bn1)
    dropout1 = tf.layers.dropout(relu1, 0.3, training=training)

    linear2 = tf.layers.dense(dropout1, 1024)
    bn2 = tf.layers.batch_normalization(linear2, training=training)
    relu2 = tf.nn.relu(bn2)
    dropout2 = tf.layers.dropout(relu2, 0.3, training=training)

    add = tf.add(inputs, dropout2)
    return add


def _xception_block(inputs, training: bool):
    s_conv1 = tf.layers.separable_conv2d(inputs, 144, [3, 3], padding='same')
    bn1 = tf.layers.batch_normalization(s_conv1, training=training)
    relu1 = tf.nn.relu(bn1)

    s_conv2 = tf.layers.separable_conv2d(relu1, 144, [3, 3], padding='same')
    bn2 = tf.layers.batch_normalization(s_conv2, training=training)
    maxpool2 = tf.layers.max_pooling2d(bn2, [3, 3], 2, padding='same')

    s_conv_skip = tf.layers.conv2d(inputs, 144, [1, 1], 2)
    bn_skip = tf.layers.batch_normalization(s_conv_skip, training=training)
    
    add = tf.add(bn_skip, maxpool2)
    return add


def build_discriminator(inputs):
    with tf.variable_scope('discriminator'):
        batch_size = tf.shape(inputs)[0]
        # Reshape from (batch, 24, 3) to (batch * 24, 3) 
        # 23 + 1 is num_joints + global rotation
        inputs_rot_mat = rodrigues_batch(tf.reshape(inputs, [-1, 3]))
        # Reshape from (batch * 24, 3, 3) to (batch, 24, 3, 3)
        inputs_rot_mat = tf.reshape(inputs_rot_mat, [batch_size, 24, 3, 3])
        inputs_rot_mat = tf.transpose(inputs_rot_mat, [0, 2, 3, 1])
        init_conv = tf.layers.conv2d(inputs_rot_mat, 1024, [3, 3])
        init_relu = tf.nn.relu(init_conv)

        flatten = tf.layers.flatten(init_relu)

        linear1 = tf.layers.dense(flatten, 1024)
        relu1 = tf.nn.relu(linear1)

        linear2 = tf.layers.dense(relu1, 1024)
        relu2 = tf.nn.relu(linear2)

        linear3 = tf.layers.dense(relu2, 1024)
        relu3 = tf.nn.relu(linear3)

        out = tf.layers.dense(relu3, 72)

        return out
