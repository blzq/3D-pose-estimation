# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from . import config
from tf_mesh_renderer import mesh_renderer
from tf_perspective_projection import project
import tf_pose.common


def render_mesh_verts_cam(verts, cam_pos, cam_rot, cam_f, faces,
                          vert_faces=None, lights=None):
    batch_size = tf.shape(verts)[0]

    if cam_pos.get_shape().as_list() == [3]:
        cam_pos = tf.reshape(tf.tile(cam_pos, [batch_size]), [batch_size, 3])
    if cam_rot.get_shape().as_list() == [3]:
        cam_rot = tf.reshape(tf.tile(cam_rot, [batch_size]), [batch_size, 3])
    cam_rot_mat = project.rodrigues_batch(cam_rot)

    origin_forward = tf.tile(tf.constant([0.0, 0.0, 1.0]), [batch_size])
    origin_forward = tf.reshape(origin_forward, [batch_size, 3, 1])
    cam_lookat = tf.matmul(cam_rot_mat, origin_forward)
    cam_lookat = tf.squeeze(cam_lookat)
    cam_lookat = tf.reshape(cam_lookat, cam_pos.shape)
    cam_lookat = cam_pos + cam_lookat

    origin_up = tf.tile(tf.constant([0.0, 1.0, 0.0]), [batch_size])
    origin_up = tf.reshape(origin_up, [batch_size, 3, 1])
    cam_up = tf.matmul(cam_rot_mat, origin_up)
    cam_up = tf.squeeze(cam_up)
    cam_up = tf.reshape(cam_up, cam_pos.shape)

    diffuse = tf.ones_like(verts, dtype=tf.float32)
    if vert_faces is None or lights is None:
        lights = tf.zeros([batch_size, 1, 3])
        normals = tf.zeros_like(verts)
    else:
        if lights.get_shape().as_list() == [3]:
            lights = tf.reshape(tf.tile(lights, [batch_size]),
                                [batch_size, -1, 3])
        normals = normals_from_mesh(verts, faces, vert_faces)
    light_intensities = tf.ones_like(lights, dtype=tf.float32)
    img_height, img_width = config.input_img_size

    rendered = mesh_renderer.mesh_renderer(
        verts, faces, normals, diffuse, cam_pos, cam_lookat, cam_up,
        lights, light_intensities, img_width, img_height,
        specular_colors=None, shininess_coefficients=None, ambient_color=None,
        fov_y=cam_f, near_clip=0.01, far_clip=100.0)

    if lights is None or vert_faces is None:
        rendered = rendered[:, :, :, 3, tf.newaxis]  # alpha ch: silhouette
    else:
        rendered = tf.reduce_mean(rendered[:, :, :, :3], axis=3, keepdims=True)

    return rendered


def get_camera_normal_plane(cam_pos, cam_rot):
    # Get the camera normal plane and offset from origin (n and d in p . n = d)
    batch_size = tf.shape(cam_pos)[0]
    cam_rot_mat = project.rodrigues_batch(cam_rot)

    origin_forward = tf.tile(tf.constant([0.0, 0.0, 1.0]), [batch_size])
    origin_forward = tf.reshape(origin_forward, [batch_size, 3, 1])
    cam_normal = tf.matmul(cam_rot_mat, origin_forward)
    cam_normal = tf.squeeze(cam_normal)
    cam_normal = tf.nn.l2_normalize(cam_normal, axis=1)

    # plane equation: p . n = d where p = cam_pos, n = cam_normal
    cam_plane_d = tf.reduce_sum(cam_pos * cam_normal, axis=1)
    return cam_normal, cam_plane_d


def get_2d_points_in_3d(points_2d, cam_rot, cam_f, img_dim):
    # Get the 2D projected points as 3D points lying on the camera image plane
    rescaled_2d = config.ss * (points_2d - img_dim / 2) / img_dim[1]
    fl_broadcast = tf.reshape(cam_f, [-1, 1, 1])
    fl_broadcast = tf.tile(fl_broadcast, [1, 24, 1])
    points_2d_in_3d = tf.concat([rescaled_2d, fl_broadcast], axis=2)
    cam_rot_mat = project.rodrigues_batch(cam_rot)
    cam_rot_mat = tf.reshape(cam_rot_mat, [-1, 1, 3, 3])
    cam_rot_mat = tf.tile(cam_rot_mat, [1, config.n_joints_smpl, 1, 1])
    points_2d_in_3d = points_2d_in_3d[:, :, :, tf.newaxis]
    points_2d_in_3d = tf.matmul(cam_rot_mat, points_2d_in_3d)
    points_2d_in_3d = tf.squeeze(points_2d_in_3d)
    return points_2d_in_3d


def normals_from_mesh(vertices, faces, vertex_faces):
    """ Compute unit normals given mesh vertices and faces
    Args:
        vertices: [batch, vertex_count, 3]
        faces: [triangle_count, 3]
        vertex_faces: [vertex_count, max_vertex_order] in the format of
                      adjacency list padded with invalid (too large) indices
                      to form a matrix
    Returns:
        normals: [batch, vertex_count, 3]
    """
    v1_indices, v2_indices, v3_indices = faces[:, 0], faces[:, 1], faces[:, 2]

    v1 = tf.gather(vertices, v1_indices, axis=1)
    v2 = tf.gather(vertices, v2_indices, axis=1)
    v3 = tf.gather(vertices, v3_indices, axis=1)

    face_normals = tf.cross(v2 - v1, v3 - v1)
    face_normals = tf.nn.l2_normalize(face_normals, axis=1)

    normals = tf.gather(face_normals, vertex_faces, axis=1)
    normals = tf.reduce_sum(normals, axis=2)
    normals = tf.nn.l2_normalize(normals, axis=2)

    # normals should point outward
    centered_vertices = vertices - tf.reduce_mean(vertices,
                                                  axis=1, keepdims=True)
    s = tf.reduce_sum(centered_vertices * normals, axis=1)

    count_s_greater_0 = tf.count_nonzero(tf.greater(s, 0), axis=1)
    count_s_less_0 = tf.count_nonzero(tf.less(s, 0), axis=1)

    sign = 2 * tf.cast(count_s_greater_0 > count_s_less_0, tf.float32) - 1
    normals = normals * tf.reshape(sign, [-1, 1, 1])

    return normals


def vertex_faces_from_face_verts(faces):
    """ From an array of faces specifying the vertices that each face contains
    of shape [n_faces, 3], get the faces corresponding to each vertex in shape
    [n_vertices, ...] in adjacency list form. Each list is then padded with
    an out of bounds (invalid) vertex index to form the array. This is since
    the GPU version of tf.gather will return 0s for out-of-bounds indices."""
    n_vertices = np.amax(faces) + 1
    vertex_faces = [ [] for _ in range(n_vertices) ]

    for face_idx, face in enumerate(faces):
        vertex_faces[face[0]].append(face_idx)
        vertex_faces[face[1]].append(face_idx)
        vertex_faces[face[2]].append(face_idx)

    vertex_orders = list(map(len, vertex_faces))
    max_vertex_order = max(vertex_orders)
    # Pad the adjacency list with out of bounds values
    vertex_faces = np.array([vf + [n_vertices] * (max_vertex_order - vo)
                            for vf, vo in zip(vertex_faces, vertex_orders)])
    return vertex_faces


def rotate_global_pose(thetas, zrot):
    # In SURREAL, the global rotation is such that the person's vertical
    # is aligned with the z axis. Make it so the person's vertical is aligned
    # with the y axis
    batch_size = tf.shape(thetas)[0]

    turn_x = tf.constant([-np.pi / 2, 0.0, 0.0])
    turn_x_batch = tf.reshape(tf.tile(turn_x, [batch_size]), [batch_size, 3])
    zeros = tf.zeros(batch_size)
    turn_y_batch = tf.stack([zeros, zrot + np.pi / 2, zeros], axis=-1)
    turn_both_batch = add_axis_angle_rotations(turn_y_batch, turn_x_batch)

    global_rot_vec = add_axis_angle_rotations(turn_both_batch, thetas[:, :3])

    thetas = tf.concat([global_rot_vec, thetas[:, 3:]], axis=1)
    return thetas


def add_axis_angle_rotations(rv1, rv2):
    # Given angle*axis rotation vectors a*l and b*m, result angle*axis: c*n
    # cos(c/2) = cos(a/2)cos(b/2) - sin(a/2)sin(b/2) (l . m)
    # sin(c/2) (n) =
    #   sin(a/2)cos(b/2) (l) + cos(a/2)sin(b/2) (m) + sin(a/2)sin(b/2) (l x m)
    a = tf.norm(rv1, axis=1, keepdims=True)
    b = tf.norm(rv2, axis=1, keepdims=True)

    l = rv1 / a
    m = rv2 / b
    cos_half_c = tf.cos(a/2) * tf.cos(b/2) - (
        tf.sin(a/2) * tf.sin(b/2) *
        tf.reduce_sum(l * m, axis=1, keepdims=True))
    sin_half_c_n = tf.sin(a/2) * tf.cos(b/2) * l + (
        tf.cos(a/2) * tf.sin(b/2) * m +
        tf.sin(a/2) * tf.sin(b/2) * tf.cross(l, m))

    half_c = tf.acos(cos_half_c)
    n = sin_half_c_n / tf.sin(half_c)
    result = n * half_c * 2

    result = tf.where(tf.equal(tf.squeeze(a), 0.0), rv2, result)
    result = tf.where(tf.equal(tf.squeeze(b), 0.0), rv1, result)
    return result


def soft_argmax(heatmaps):
    # https://arxiv.org/pdf/1603.09114.pdf - equation 7
    # use softmax(heatmaps) * indices as differentiable replacement for argmax
    strength = 100.0  # Tunable parameter for softmax; higher -> sharper peak
    shape = tf.shape(heatmaps)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
    x_ind, y_ind = tf.meshgrid(tf.range(w), tf.range(h))
    y_ind = tf.cast(y_ind, tf.float32)
    x_ind = tf.cast(x_ind, tf.float32)

    softmax = tf.nn.softmax(tf.reshape(strength * heatmaps, [b, h * w, c]),
                            axis=1)
    softmax = tf.reshape(softmax, [b, h, w, c])

    soft_argmax_y = softmax * tf.reshape(y_ind, [1, h, w, 1])
    soft_argmax_x = softmax * tf.reshape(x_ind, [1, h, w, 1])

    y_locations = tf.reduce_sum(soft_argmax_y, axis=[1, 2])
    x_locations = tf.reduce_sum(soft_argmax_x, axis=[1, 2])
    maxes = tf.reduce_max(heatmaps, axis=[1, 2])

    return tf.stack([y_locations, x_locations], axis=2), maxes


def soft_argmax_rescaled(heatmaps):
    img_dim = tf.cast(tf.shape(heatmaps)[1:3], tf.float32)
    locations, maxes = soft_argmax(heatmaps)

    # Move centre of image to (0, 0)
    half_img_dim = img_dim / 2
    locations = locations - half_img_dim[tf.newaxis]
    # Scale detection locations by shorter side length
    img_side_len = tf.minimum(img_dim[0], img_dim[1])
    locations = locations / img_side_len[tf.newaxis]

    # Maybe don't want to do this part because information for camera is lost
    # Normalize centre of person as middle of left and right hips
    # rhip_idx = tf_pose.common.CocoPart.RHip.value
    # lhip_idx = tf_pose.common.CocoPart.LHip.value
    # centres = (locations[:, rhip_idx, :] + locations[:, lhip_idx, :]) / 2
    # locations = locations - centres[:, tf.newaxis]
    # Normalize joint locations to [-1, 1] in x and y
    # max_extents = tf.reduce_max(tf.abs(locations), axis=[1, 2], keepdims=True)
    # locations = locations / max_extents

    return tf.concat([locations, maxes[..., tf.newaxis]], axis=2)


def gaussian_blur(img, kernel_size=13, sigma=7):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d_native(img, gaussian_kernel, [1, 1, 1, 1],
                                         padding='SAME', data_format='NHWC')
