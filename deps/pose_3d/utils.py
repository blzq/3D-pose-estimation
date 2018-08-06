# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pkg_resources

from . import config
from tf_mesh_renderer import mesh_renderer
import tf_smpl
from tf_perspective_projection import project


def render_mesh_verts_cam(verts, cam_pos, cam_rot, cam_f, faces, lights=None):
    batch_size = tf.shape(verts)[0]

    # TODO: Calculate this properly for real colour
    normals = tf.nn.l2_normalize(verts, axis=1)

    if cam_pos.get_shape().as_list() == [3]:
        cam_pos = tf.reshape(tf.tile(cam_pos, [batch_size]), [batch_size, 3])
    if cam_rot.get_shape().as_list() == [3]:
        cam_rot = tf.reshape(tf.tile(cam_rot, [batch_size]), [batch_size, 3])
    cam_rot_mat = project.rodrigues_batch(cam_rot)

    origin_forward = tf.tile(tf.constant([0.0, 0.0, 1.0]), [batch_size])
    origin_forward = tf.reshape(origin_forward, [batch_size, 3, 1])
    cam_lookat = tf.matmul(cam_rot_mat, origin_forward)
    cam_lookat = tf.squeeze(cam_lookat)
    cam_lookat = cam_pos + cam_lookat
    cam_lookat.set_shape(cam_pos.shape)

    origin_up = tf.tile(tf.constant([0.0, 1.0, 0.0]), [batch_size])
    origin_up = tf.reshape(origin_up, [batch_size, 3, 1])
    cam_up = tf.matmul(cam_rot_mat, origin_up)
    cam_up = tf.squeeze(cam_up)
    cam_up.set_shape(cam_pos.shape)

    diffuse = tf.ones_like(verts, dtype=tf.float32)
    if lights is None:
        light_pos = tf.zeros([batch_size, 1, 3])
    else:
        light_pos = tf.reshape(tf.tile(lights, [batch_size, 1]),
                               [batch_size, -1, 3])
    light_intensities = tf.ones_like(light_pos, dtype=tf.float32)
    img_height, img_width = config.input_img_size

    rendered = mesh_renderer.mesh_renderer(
        verts, faces, normals, diffuse, cam_pos, cam_lookat, cam_up,
        light_pos, light_intensities, img_width, img_height,
        specular_colors=None, shininess_coefficients=None, ambient_color=None,
        fov_y=cam_f, near_clip=0.01, far_clip=10.0)

    if lights is None:
        rendered = rendered[:, :, :, 3, tf.newaxis]
    else:
        rendered = tf.reduce_mean(rendered[:, :, :, :3], axis=3, keepdims=True)

    return rendered


def rotate_global_pose(thetas):
    batch_size = tf.shape(thetas)[0]
    global_rot = thetas[:, :3]
    global_rot_mat = project.rodrigues_batch(global_rot)
    # turn_x = np.array([[1.0,  0.0, 0.0],
    #                    [0.0,  0.0, 1.0],
    #                    [0.0, -1.0, 0.0]])
    # turn_y = tf.constant([[0.0, 0.0, -1.0],
    #                       [0.0, 1.0,  0.0],
    #                       [1.0, 0.0,  0.0]])
    turn_b = tf.constant([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])
    turn_b = tf.reshape(tf.tile(turn_b, [batch_size, 1]), [batch_size, 3, 3])
    global_rot_mat = tf.matmul(turn_b, global_rot_mat)

    # https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_axis
    global_rv_1 = global_rot_mat[:, 2, 1] - global_rot_mat[:, 1, 2]
    global_rv_2 = global_rot_mat[:, 0, 2] - global_rot_mat[:, 2, 0]
    global_rv_3 = global_rot_mat[:, 1, 0] - global_rot_mat[:, 0, 1]

    global_rot_vec = tf.stack([global_rv_1, global_rv_2, global_rv_3], axis=1)
    global_rot_vec = tf.nn.l2_normalize(global_rot_vec, axis=1)
    global_rot_vec = global_rot_vec * tf.acos(
        (tf.trace(global_rot_mat) - 1) / 2)[:, tf.newaxis]
    thetas = tf.concat([global_rot_vec, thetas[:, 3:]], axis=1)

    return thetas
