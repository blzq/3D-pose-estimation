#!/usr/bin/python3
# -*- coding: utf-8 -*-

import __init__

import sys
import os
import glob

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tf_pose.estimator import TfPoseEstimator as OpPoseEstimator
from tf_pose.networks import get_graph_path

from pose_3d.pose_model_3d import PoseModel3d
from pose_3d import data_helpers
from pose_3d import config, utils

from tf_smpl.batch_smpl import SMPL

SAVER_PATH = '/home/ben/tensorflow_logs/3d_pose/ckpts/3d_pose.ckpt'
SUMMARY_DIR = '/home/ben/tensorflow_logs/3d_pose/'


def main():
    images_path = os.path.join(__init__.project_path, 'data', 'images')
    in_im = cv2.imread(os.path.join(images_path, 'coco3.png'))
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

    print(out_vals)
    out_vals[:, :3] = 0

    smpl_dir = os.path.join(__init__.project_path,
                            'data', 'SMPL_model', 'models_numpy')
    smpl_model_path = os.path.join(smpl_dir, 'model_neutral_np.pkl')

    smpl = SMPL(smpl_model_path)
    beta = tf.zeros([1, 10])
    pose = tf.constant(out_vals[:, :72])

    verts, _, _ = smpl(beta, pose, get_skin=True)
    verts = verts[0]

    result = tf.Session().run(verts)

    dirpath = os.path.dirname(os.path.realpath(__file__))
    faces = np.load(os.path.join(dirpath, '..', 'deps', 'tf_smpl',
                                 'smpl_faces.npy'))

    outmesh_path = os.path.join(dirpath, 'smpl_tf.obj')
    with open(outmesh_path, 'w') as fp:
        for v in result:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    op_out_im = OpPoseEstimator.draw_humans(in_im, humans, imgcopy=True)
    plt.subplot(121)
    plt.imshow(np.sum(heatmaps[0][:, :, :config.n_joints], axis=2),
               cmap='gray')
    plt.subplot(122)
    plt.imshow(op_out_im)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
