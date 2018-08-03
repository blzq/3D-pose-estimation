# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pkg_resources

from . import config
from tf_mesh_renderer import mesh_renderer
import tf_smpl
from tf_perspective_projection import project


def render_mesh_verts_cam(verts, cam_pos, cam_rot, cam_f, batch_size):
    faces_path = pkg_resources.resource_filename(
        tf_smpl.__name__, 'smpl_faces.npy')
    faces = np.load(faces_path)
    faces = tf.constant(faces, dtype=tf.int32)

    # verts.set_shape([batch_size, 6890, 3])
    batch_size = tf.shape(verts)[0]
    # print(batch_size)

    # Calculate this properly for real colour
    normals = tf.zeros_like(verts)

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
    light_pos = tf.zeros([batch_size, 1, 3])
    light_intensities = tf.zeros([batch_size, 1, 3])
    img_height, img_width = config.input_img_size

    rendered = mesh_renderer.mesh_renderer(
        verts, faces, normals, diffuse, cam_pos, cam_lookat, cam_up,
        light_pos, light_intensities, img_width, img_height,
        specular_colors=None, shininess_coefficients=None, ambient_color=None,
        fov_y=cam_f, near_clip=0.01, far_clip=10.0)
    
    return rendered[:, :, :, :3]

    

    

    