#!/usr/bin/python3
# -*- coding: utf-8 -*-

import __init__

import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from tf_pose.estimator import TfPoseEstimator as OpPoseEstimator
from tf_pose.networks import get_graph_path

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
images_dir = os.path.join(project_dir, 'data', 'images')


def main():
    in_im = cv2.imread(os.path.join(images_dir, 'train_image.jpg'))
    if in_im is None:
        print("Image at path given doesn't exist")
        return
    in_im = cv2.cvtColor(in_im, cv2.COLOR_BGR2RGB)
    estimator = OpPoseEstimator(get_graph_path('cmu'))
    humans = estimator.inference(in_im, upsample_size=8.0)
    fig = plt.figure()
    for i in range(estimator.heatMat.shape[2]):
        fig.add_subplot(estimator.heatMat.shape[2] // 5 + 1, 5, i+1)
        plt.imshow(estimator.heatMat[:, :, i])
    plt.show()
    fig = plt.figure()
    for i in range(estimator.pafMat.shape[2]):
        fig.add_subplot(estimator.pafMat.shape[2] // 6 + 1, 6, i+1)
        plt.imshow(estimator.pafMat[:, :, i])
    plt.show()
    return


if __name__ == '__main__':
    sys.exit(main())