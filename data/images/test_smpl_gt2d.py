import cv2
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('joints_vis.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    info_dict = scipy.io.loadmat('01_11_c0001_info.mat')
    joints2d = np.transpose(info_dict['joints2D'], (2, 1, 0))
    joints2d = joints2d[0]
    print("SMPL no. of joints: " + str(len(joints2d)))
    for idx, joint in enumerate(joints2d):
        for idy, joint in enumerate(joints2d):
            col = [255, 255, 255] if idx != idy else [255, 0, 255]
            j = (int(joint[0]), int(joint[1]))
            cv2.circle(img, j, 3, col)

            plt.imsave("joints/{:02}.jpg".format(idx), img)
    