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
from pose_3d.data_helpers import *

from tf_smpl.batch_smpl import SMPL

SAVER_PATH = '/home/ben/tensorflow_ckpts/3d_pose'


def main():
    images_path = os.path.join(__init__.project_path, 'data', 'images')
    in_im = cv2.imread(os.path.join(images_path, 'train_image.jpg'))
    in_im = cv2.cvtColor(in_im, cv2.COLOR_BGR2RGB)
    in_im = cv2.resize(in_im, dsize=(160, 120),
               interpolation=cv2.INTER_AREA)

    estimator = OpPoseEstimator(get_graph_path('cmu'))
    
    humans = estimator.inference(in_im, upsample_size=4.0)
    heatmaps = estimator.heatMat
    heatmaps = heatmaps[np.newaxis]  # add "batch" axis

    inputs = np.concatenate([heatmaps, in_im[np.newaxis]], axis=3)

    pm_3d = PoseModel3d((None, 120, 160, 22),
                        training=False,
                        summary_dir='/tmp/null',
                        saver_path=SAVER_PATH,
                        restore_model=True)
    out_vals = pm_3d.estimate(inputs)

    print(out_vals)

    smpl_dir = os.path.join(__init__.project_path,
                            'data', 'SMPL_model', 'models_numpy')
    smpl_model_path = os.path.join(smpl_dir, 'model_neutral_np.pkl')

    smpl = SMPL(smpl_model_path)
    beta = tf.zeros([1, 10])
    pose = tf.constant(out_vals)
    
    verts, _, _ = smpl(beta, pose, get_skin=True)
    verts = verts[0]

    with tf.Session() as sess:
        result = sess.run(verts)

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
    plt.imshow(op_out_im)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())










