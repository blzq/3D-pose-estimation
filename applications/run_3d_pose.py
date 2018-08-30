#!/usr/bin/python3
# -*- coding: utf-8 -*-

import __init__

import sys
import os
import glob
import pkg_resources

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tf_pose.estimator import TfPoseEstimator as OpPoseEstimator
from tf_pose.networks import get_graph_path

from pose_3d.pose_model_3d import PoseModel3d
from pose_3d import data_helpers
from pose_3d import config, utils

import tf_smpl
from tf_smpl.batch_smpl import SMPL


SAVER_PATH = '/home/ben/tensorflow_logs/3d_pose/ckpts/3d_pose.ckpt'
SUMMARY_DIR = '/home/ben/tensorflow_logs/3d_pose/'


def main(in_filename):
    images_path = os.path.join(__init__.project_path, 'data', 'images')
    in_im = cv2.imread(os.path.join(images_path, in_filename))
    in_im = cv2.cvtColor(in_im, cv2.COLOR_BGR2RGB)
    img_size = config.input_img_size
    expect_aspect = img_size[1] / img_size[0]
    in_shape = in_im.shape
    in_aspect = in_shape[1] / in_shape[0]
    if in_aspect != expect_aspect:
        if in_im.shape[1] >= in_im.shape[0] * expect_aspect:
            diff = int(in_im.shape[1] - in_im.shape[0] * expect_aspect)
            pad_im = cv2.copyMakeBorder(in_im, 0, diff, 0, 0,
                                        cv2.BORDER_CONSTANT, None, 0)
        else:
            diff = int(in_im.shape[0] * expect_aspect - in_im.shape[1])
            pad_im = cv2.copyMakeBorder(in_im, 0, 0, 0, diff,
                                        cv2.BORDER_CONSTANT, None, 0)
        in_im = pad_im

    in_im = cv2.resize(in_im, dsize=(img_size[1], img_size[0]),
                       interpolation=cv2.INTER_AREA)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # pylint: disable=no-member
    with tf.Graph().as_default():
        estimator = OpPoseEstimator(get_graph_path('cmu'),
                                    target_size=(img_size[1]*2, img_size[0]*2),
                                    tf_config=tfconfig)
    humans = estimator.inference(in_im,
                                 resize_to_default=True, upsample_size=4)
    heatmaps = estimator.heatMat[:, :, :config.n_joints]
    heatmaps = data_helpers.suppress_non_largest_human(humans,
                                                       heatmaps, img_size)
    heatmaps = heatmaps[np.newaxis]  # add "batch" axis

    in_im_3d = cv2.normalize(in_im, None, 0, 1, cv2.NORM_MINMAX)
    inputs = np.concatenate([heatmaps, in_im_3d[np.newaxis]], axis=3)

    # Visualise argmaxs
    # input_locs = tf.Session().run(utils.soft_argmax_rescaled(heatmaps))
    # input_locs = input_locs[0]
    # plt.scatter(input_locs[:, 1], input_locs[:, 0])
    # plt.gca().invert_yaxis()
    # plt.show()
    # for input_loc in input_locs[0]:
    #     y, x, _ = (input_loc * 240 + np.array([120, 160, 0])).astype(np.int32)
    #     in_im = cv2.circle(in_im, (x, y), 3, (0, 255, 255))
    # plt.imshow(in_im)
    # plt.show()
    # exit()

    # with different graph so checkpoint is restored correctly
    pm_graph = tf.Graph()

    pm_3d = PoseModel3d((None, 240, 320, 3 + config.n_joints),
                        pm_graph,
                        mode='test',
                        summary_dir=SUMMARY_DIR,
                        saver_path=SAVER_PATH,
                        restore_model=True)
    out_vals = pm_3d.estimate(inputs)

    smpl_dir = os.path.join(__init__.project_path,
                            'data', 'SMPL_model', 'models_numpy')
    smpl_model_path = os.path.join(smpl_dir, 'model_neutral_np.pkl')
    faces_path = pkg_resources.resource_filename(tf_smpl.__name__,
                                                 'smpl_faces.npy')
    faces = np.load(faces_path)

    smpl = SMPL(smpl_model_path)
    beta = tf.zeros([1, 10])
    pose = tf.constant(out_vals[:, :72])

    cam_pos = tf.constant(out_vals[:, 72:75])
    cam_rot = tf.constant(out_vals[:, 75:78])
    cam_f = tf.tile([config.fl], [out_vals.shape[0]])

    verts, _, _ = smpl(beta, pose, get_skin=True)

    vert_faces = utils.vertex_faces_from_face_verts(faces)
    mesh_img = utils.render_mesh_verts_cam(
        verts, cam_pos, cam_rot, tf.atan2(config.ss / 2, cam_f) * 360 / np.pi,
        tf.constant(faces, dtype=tf.int32), vert_faces,
        cam_pos[:, tf.newaxis, :])

    verts_eval, mesh_img_eval = tf.Session().run((verts[0], mesh_img[0]))

    dirpath = os.path.dirname(os.path.realpath(__file__))
    outmesh_path = os.path.join(dirpath, 'smpl_tf.obj')
    with open(outmesh_path, 'w') as fp:
        for v in verts_eval:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    op_out_im = OpPoseEstimator.draw_humans(in_im, humans, imgcopy=True)
    plt.subplot(131)
    plt.imshow(np.sum(heatmaps[0][:, :, :config.n_joints], axis=2),
               cmap='gray')
    plt.subplot(132)
    plt.imshow(op_out_im)
    plt.subplot(133)
    print(mesh_img_eval.shape)
    plt.imshow(np.squeeze(mesh_img_eval), cmap='gray')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 run_3d_pose.py <path-to-input-image>")
    sys.exit(main(sys.argv[1]))
