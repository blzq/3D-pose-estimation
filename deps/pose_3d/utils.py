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
    # In SURREAL, the global rotation is such that the person's vertical
    # is aligned with the z axis. Make it so the person's vertical is aligned
    # with the y axis.
    batch_size = tf.shape(thetas)[0]

    turn_x = tf.constant([np.pi / 2, 0.0, 0.0])
    turn_x_batch = tf.reshape(tf.tile(turn_x, [batch_size]), [batch_size, 3])

    global_rot_vec = add_axis_angle_rotations(turn_x_batch, thetas[:, :3])

    # global_rot = thetas[:, :3]
    # global_rot_mat = project.rodrigues_batch(global_rot)
    # turn_x = tf.constant([[1.0,  0.0, 0.0],
    #                       [0.0,  0.0, 1.0],
    #                       [0.0, -1.0, 0.0]])
    # turn_y = tf.constant([[0.0, 0.0, -1.0],
    #                       [0.0, 1.0,  0.0],
    #                       [1.0, 0.0,  0.0]])
    # turn_both = tf.constant([[0.0, 1.0, 0.0],
    #                       [0.0, 0.0, 1.0],
    #                       [1.0, 0.0, 0.0]])
    # turn_b = tf.reshape(tf.tile(turn_b, [batch_size, 1]), [batch_size, 3, 3])
    # global_rot_mat = tf.matmul(turn_b, global_rot_mat)

    # Rotation matrix to axis-angle
    # https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_axis
    # Note the case where rotation angle = pi (R = Rt) is treated as angle = 0
    # global_rv_1 = global_rot_mat[:, 2, 1] - global_rot_mat[:, 1, 2]
    # global_rv_2 = global_rot_mat[:, 0, 2] - global_rot_mat[:, 2, 0]
    # global_rv_3 = global_rot_mat[:, 1, 0] - global_rot_mat[:, 0, 1]
    # global_rot_vec = tf.stack([global_rv_1, global_rv_2, global_rv_3], axis=1)
    # global_rot_vec_norm = tf.norm(global_rot_vec, axis=1, keepdims=True)
    # is_zero = tf.equal(tf.squeeze(global_rot_vec_norm), 0.0)
    # global_rot_vec = global_rot_vec / global_rot_vec_norm
    # global_rot_vec = tf.where(is_zero,
    #                           tf.zeros_like(global_rot_vec), global_rot_vec)

    # global_rot_vec = global_rot_vec * tf.asin(global_rot_vec_norm / 2)

    thetas = tf.concat([global_rot_vec, thetas[:, 3:]], axis=1)
    return thetas


def add_axis_angle_rotations(rv1, rv2):
    # given angle*axis rotation vectors a*l and b*m, result angle*axis: c*n
    # cos(c/2) = cos(a/2)cos(b/2) - sin(a/2)sin(b/2) (l . m)
    # sin(c/2) (n) = 
    #   sin(a/2)cos(b/2) (l) + cos(a/2)sin(b/2) (m) + sin(a/2)sin(b/2) (l x m)
    a = tf.norm(rv1, axis=1)
    b = tf.norm(rv2, axis=1)

    l = tf.nn.l2_normalize(rv1)
    m = tf.nn.l2_normalize(rv2)

    cos_half_c = tf.cos(a/2) * tf.cos(b/2) - (
        tf.sin(a/2) * tf.sin(b/2) * tf.matmul(l, m, transpose_b=True))
    sin_half_c_n = tf.sin(a/2) * tf.cos(b/2) * l + (
        tf.cos(a/2) * tf.sin(b/2) * m + 
        tf.sin(a/2) * tf.sin(b/2) * tf.cross(l, m))

    half_c = tf.acos(cos_half_c)
    n = sin_half_c_n / tf.sin(half_c)
    return n * half_c * 2
