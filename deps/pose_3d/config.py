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
# 1: RHip
# 2: LHip
# 3: BackLower (Stomach)
# 4: RKnee
# 5: LKnee
# 6: BackCentre
# 7: RAnkle
# 8: LAnkle
# 9: BackUpper
# 10: RFoot
# 11: LFoot
# 12: NeckLower (Head)
# 13: RShoulderInner
# 14: LShoulderInner
# 15: NeckUpper
# 16: RShoulderOuter
# 17: LShoulderInner
# 18: RElbow
# 19: LElbow
# 20: RWrist
# 21: LWrist
# 22: RHand
# 23: LHand