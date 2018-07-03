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
    in_im = cv2.imread(os.path.join(images_dir, 'test_image.jpg'))
    in_im = cv2.cvtColor(in_im, cv2.COLOR_BGR2RGB)
    estimator = OpPoseEstimator(get_graph_path('cmu'))
    humans = estimator.inference(in_im, upsample_size=8.0)
    out_im = OpPoseEstimator.draw_humans(in_im, humans, imgcopy=True)
    plt.imshow(out_im)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())