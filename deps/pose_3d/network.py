# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

from tf_perspective_projection.project import rodrigues_batch
from tf_pose.common import CocoPart

from . import utils
from . import config


def build_model(inputs, training: bool):
    with tf.variable_scope('encoder'):
        with tf.variable_scope("preprocess_heatmaps"):
            inputs = tf.check_numerics(inputs, "inputs not finite")
            input_rgb = inputs[:, :, :, config.n_joints:]
            input_heatmaps = inputs[:, :, :, :config.n_joints]
            input_heatmaps = utils.gaussian_blur(input_heatmaps)
            inputs = tf.concat([input_heatmaps, input_rgb], axis=3)
            tf.summary.image('in_images', 
                             tf.reduce_sum(input_rgb, axis=3, keepdims=True),
                             max_outputs=1)

        with tf.variable_scope('init_conv'):
            in_channels = inputs.get_shape().as_list()[-1]
            init_conv1 = tf.layers.conv2d(inputs, in_channels, [3, 3])
            bn1 = tf.layers.batch_normalization(init_conv1, training=training)
            conv_relu1 = tf.nn.relu(bn1)

        with tf.variable_scope('mobilenetv2'):
            mn = _mobilenetv2(conv_relu1, training, alpha=1.1)

        with tf.variable_scope('input_locations'):
            input_locations = utils.soft_argmax_rescaled(input_heatmaps)
            locations_flat = tf.layers.flatten(input_locations)

        with tf.variable_scope('bilinear_blocks'):
            features_flat = tf.layers.flatten(mn)
            features_drop = tf.layers.dropout(features_flat, 0.2,
                                              training=training)
            in_concat = tf.concat([features_drop, locations_flat], axis=1)
            l_units = 1536
            in_dense = tf.layers.dense(in_concat, l_units)
            in_bn = tf.layers.batch_normalization(in_dense, training=training)
            in_relu = tf.nn.relu(in_bn)
            bl1 = _bilinear_res_block(in_relu, l_units, training,
                                      input_activation=False)
            bl2 = _bilinear_res_block(bl1, l_units, training)
            bl3 = _bilinear_res_block(bl2, l_units, training)
            bl4 = _bilinear_res_block(bl3, l_units, training)

        with tf.variable_scope('camera_blocks'):
            cam_units = 256
            cam_d = tf.layers.dense(locations_flat, cam_units)
            cam_bn = tf.layers.batch_normalization(cam_d, training=training)
            cam_relu = tf.nn.relu(cam_bn)
            bl1_cam = _bilinear_res_block(cam_relu, cam_units, training,
                                          input_activation=False)
            bl2_cam = _bilinear_res_block(bl1_cam, cam_units, training)
            bl3_cam = _bilinear_res_block(bl2_cam, cam_units, training)
            bl4_cam = _bilinear_res_block(bl3_cam, cam_units, training)

        with tf.variable_scope('out_fc'):
            out_pose = tf.layers.dense(bl4, 72) # 24*3 rotations
            out_cam = tf.layers.dense(bl4_cam, 7) # f x 1, t x 3, r x 3
            out = tf.concat([out_pose, out_cam], axis=1)

    return out


def _bilinear_res_block(inputs, units, training: bool, input_activation=True):
    if input_activation:
        in_bn = tf.layers.batch_normalization(inputs, training=training)
        in_relu = tf.nn.relu(in_bn)
        linear_in = tf.layers.dropout(in_relu, 0.2, training=training)
    else:
        linear_in = inputs
    linear1 = tf.layers.dense(linear_in, units)

    bn1 = tf.layers.batch_normalization(linear1, training=training)
    relu1 = tf.nn.relu(bn1)
    dropout1 = tf.layers.dropout(relu1, 0.2, training=training)
    linear2 = tf.layers.dense(dropout1, units)

    add = tf.add(inputs, linear2)
    return add


def _mobilenetv2(inputs, training: bool, alpha=1.4):
    init_filters = _make_divisible(32 * alpha, 8)
    init_conv2d = tf.layers.conv2d(inputs, init_filters, [3, 3], 2)
    init_bn = tf.layers.batch_normalization(init_conv2d, training=training)
    init_relu = tf.nn.relu6(init_bn)

    mn1 = _mobilenetv2_block(init_relu, 16, 1, 1, alpha, training)

    mn2 = _mobilenetv2_block(mn1, 24, 2, 6, alpha, training)
    mn3 = _mobilenetv2_block(mn2, 24, 1, 6, alpha, training)

    mn4 = _mobilenetv2_block(mn3, 32, 2, 6, alpha, training)
    mn5 = _mobilenetv2_block(mn4, 32, 1, 6, alpha, training)
    mn6 = _mobilenetv2_block(mn5, 32, 1, 6, alpha, training)

    mn7 = _mobilenetv2_block(mn6, 64, 2, 6, alpha, training)
    mn8 = _mobilenetv2_block(mn7, 64, 1, 6, alpha, training)
    mn9 = _mobilenetv2_block(mn8, 64, 1, 6, alpha, training)
    mn10 = _mobilenetv2_block(mn9, 64, 1, 6, alpha, training)

    mn11 = _mobilenetv2_block(mn10, 96, 1, 6, alpha, training)
    mn12 = _mobilenetv2_block(mn11, 96, 1, 6, alpha, training)
    mn13 = _mobilenetv2_block(mn12, 96, 1, 6, alpha, training)

    mn14 = _mobilenetv2_block(mn13, 160, 2, 6, alpha, training)
    mn15 = _mobilenetv2_block(mn14, 160, 1, 6, alpha, training)
    mn16 = _mobilenetv2_block(mn15, 160, 1, 6, alpha, training)

    mn17 = _mobilenetv2_block(mn16, 320, 1, 6, alpha, training)

    last_filters = _make_divisible(1280 * alpha, 8) if alpha > 1.0 else 1280
    last_conv = tf.layers.conv2d(mn17, last_filters, [1, 1])
    last_bn = tf.layers.batch_normalization(last_conv, training=training)
    last_relu = tf.nn.relu6(last_bn)

    pool = tf.reduce_mean(last_relu, axis=[1, 2])
    return pool


def _mobilenetv2_block(inputs, filters: int, stride: int, expansion: int,
                       alpha: float, training: bool):
    in_channels = inputs.get_shape().as_list()[-1]
    pointwise_conv_filters = _make_divisible(int(filters * alpha), 8)

    if expansion > 1:
        expand = tf.layers.conv2d(inputs, expansion * in_channels, [1, 1])
        ex_bn = tf.layers.batch_normalization(expand, training=training)
        ex_out = tf.nn.relu6(ex_bn)
    else:
        ex_out = inputs

    depthwise = keras.layers.DepthwiseConv2D([3, 3], stride,
                                             padding='same')(ex_out)
    dw_bn = tf.layers.batch_normalization(depthwise, training=training)
    dw_relu = tf.nn.relu6(dw_bn)
    pointwise = tf.layers.conv2d(dw_relu, pointwise_conv_filters, [1, 1])
    pw_bn = tf.layers.batch_normalization(pointwise, training=training)

    if in_channels == pointwise_conv_filters and stride == 1:
        out = tf.add(inputs, pw_bn)
    else:
        out = pw_bn
    return out


def _make_divisible(v, divisor: int):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_discriminator(inputs):
    with tf.variable_scope('discriminator'):
        batch_size = tf.shape(inputs)[0]
        # Reshape from (batch, 24, 3) to (batch * 24, 3)
        # 23 + 1 is num_joints + global rotation
        inputs_rot_mat = rodrigues_batch(tf.reshape(inputs, [-1, 3]))
        # Reshape from (batch * 24, 3, 3) to (batch, 24, 3, 3)
        inputs_rot_mat = tf.reshape(inputs_rot_mat, [batch_size, 24, 3, 3])
        # Transpose from (batch * 24, 3, 3) to (batch, 3, 3, 24)
        inputs_rot_mat = tf.transpose(inputs_rot_mat, [0, 2, 3, 1])
        init_conv1 = tf.layers.conv2d(inputs_rot_mat, 1024, [3, 3])
        init_relu1 = tf.nn.leaky_relu(init_conv1)

        flatten = tf.layers.flatten(init_relu1)

        linear1 = tf.layers.dense(flatten, 1024)
        relu1 = tf.nn.leaky_relu(linear1)

        linear2 = tf.layers.dense(relu1, 1024)
        relu2 = tf.nn.leaky_relu(linear2)

        linear3 = tf.layers.dense(relu2, 72)
        relu3 = tf.nn.leaky_relu(linear3)

        out = tf.layers.dense(relu3, 2)
        return out
