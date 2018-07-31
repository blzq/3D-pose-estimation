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

input_img_size = (240, 320)

# SMPL model joints in order:
# 0: CentreHip (Global rotation)
# 1: LHip
# 2: RHip
# 3: BackLower (Stomach)
# 4: LKnee
# 5: RKnee
# 6: BackCentre
# 7: LAnkle
# 8: RAnkle
# 9: BackUpper
# 10: LFoot
# 11: RFoot
# 12: NeckLower
# 13: LShoulderInner
# 14: RShoulderInner
# 15: NeckUpper
# 16: LShoulderOuter
# 17: RShoulderInner
# 18: LElbow
# 19: RElbow
# 20: LWrist
# 21: RWrist
# 22: LHand
# 23: RHand