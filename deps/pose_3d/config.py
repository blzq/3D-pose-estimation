# -*- coding: utf-8 -*-

from math import pi

pose_loss_scale = 1.0
pose_loss_direct_scale = 0.05
mesh_loss_scale = 2.5
joint_loss_scale = 10.0
reg_loss_scale = 10.0
reg_joint_limit = pi * 2.0
disc_loss_scale = 2
reproj_loss_scale = 0.000016
n_joints = 18  # COCO number of joints (not including background)
n_joints_smpl = 24  # SMPL model number of joints
cam_loss_scale = 0.000016
