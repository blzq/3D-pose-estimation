import cv2
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('train_image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    info_dict = scipy.io.loadmat('01_01_c0001_info.mat')
    joints2d = np.transpose(info_dict['joints2D'], (2, 1, 0))
    joints2d = joints2d[0]
    
    for joint in joints2d:
        j = (int(joint[0]), int(joint[1]))
        cv2.circle(img, j, 3, [255, 0, 255])

    plt.imshow(img)
    plt.show()
    