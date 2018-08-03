# -*- coding: utf-8 -*-

from math import pi

total_loss_scale = 1.0
pose_loss_scale = 1.0
pose_loss_direct_scale = 1.0
mesh_loss_scale = 1.0
joint_loss_scale = 2.0
reg_loss_scale = 5.0
reg_joint_limit = pi * 2.0
disc_loss_scale = 2
reproj_loss_scale = 0  # 0.000016
n_joints = 18  # COCO number of joints (not including background)
n_joints_smpl = 24  # SMPL model number of joints
cam_loss_scale = 0.000016

input_img_size = (240, 320)

# SMPL model joints order:
#  0: CentreHip (Global rotation)
#  1: LHip
#  2: RHip
#  3: BackLower (Stomach)
#  4: LKnee
#  5: RKnee
#  6: BackCentre
#  7: LAnkle
#  8: RAnkle
#  9: BackUpper
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

# COCO joints order (output of openpose heatmaps):
#  0: Nose = 0
#  1: Neck = 1
#  2: RShoulder = 2
#  3: RElbow = 3
#  4: RWrist = 4
#  5: LShoulder = 5
#  6: LElbow = 6
#  7: LWrist = 7
#  8: RHip = 8
#  9: RKnee = 9
# 10: RAnkle = 10
# 11: LHip = 11
# 12: LKnee = 12
# 13: LAnkle = 13
# 14: REye = 14
# 15: LEye = 15
# 16: REar = 16
# 17: LEar = 17
# 18: Background = 18

# LSP joints order:
#  0: Right ankle
#  1: Right knee
#  2: Right hip
#  3: Left hip
#  4: Left knee
#  5: Left ankle
#  6: Right wrist
#  7: Right elbow
#  8: Right shoulder
#  9: Left shoulder
# 10: Left elbow
# 11: Left wrist
# 12: Neck
# 13: Head top
