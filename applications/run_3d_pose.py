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

from smpl_numpy.smpl_np import SMPLModel

SAVER_PATH = '/home/ben/tensorflow_ckpts/3d_pose'


def main():
    images_path = os.path.join(__init__.project_path, 'data', 'images')
    in_im = cv2.imread(os.path.join(images_path, 'train_image.jpg'))
    in_im = cv2.cvtColor(in_im, cv2.COLOR_BGR2RGB)

    estimator = OpPoseEstimator(get_graph_path('cmu'))
    humans = estimator.inference(in_im, upsample_size=8.0)
    heatmaps = estimator.heatMat
    heatmaps = cv2.resize(heatmaps, dsize=(160, 120),
                          interpolation=cv2.INTER_CUBIC)
    heatmaps = heatmaps[np.newaxis]  # add "batch" axis

    pm_3d = PoseModel3d((None, 120, 160, 19),
                        training=False,
                        summary_dir='/tmp/null',
                        saver_path=SAVER_PATH,
                        restore_model=True)
    out_vals = pm_3d.estimate(heatmaps)

    smpl_dir = os.path.join(__init__.project_path,
                            'data', 'SMPL_model', 'models_numpy')
    smpl_model_path = os.path.join(smpl_dir, 'model_m.pkl')

    smpl = SMPLModel(smpl_model_path)
    beta = np.zeros(smpl.beta_shape)
    pose = out_vals
    trans = np.zeros(smpl.trans_shape)
    smpl.set_params(beta=beta, pose=pose, trans=trans)

    file_path = os.path.dirname(os.path.realpath(__file__))
    smpl.save_to_obj(os.path.join(file_path, 'out.obj'))

    op_out_im = OpPoseEstimator.draw_humans(in_im, humans, imgcopy=True)
    plt.imshow(op_out_im)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())










