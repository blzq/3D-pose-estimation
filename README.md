## 3D-pose-estimation

3D human pose estimation from 2D heatmaps and RGB image.

### Installation
``` git clone --recursive https://github.com/blzq/3D-pose-estimation.git ```

Requires `tensorflow` and `opencv`.

Requires the SMPL model. See the installation instructions in `deps/tf_smpl`.
Requires additional C++ compilation steps in `deps/tf_pose` and `deps/tf_mesh_renderer`.

### Directory structure
`applications` contains scripts to train, test, and evaluate the model, as well as scripts to generate heatmaps
(using OpenPose) for the SURREAL and Humans3.6M datasets. The Humans3.6M dataset to use is provided by 
`https://github.com/akanazawa/hmr`, and contains ground truth for SMPL parameters generated using MoSH.

`deps/pose_3d/` contains code for this project.

`pose_model_3d.py` is the main class to instantiate the model. `network.py` contains the model definition.
`data_helpers.py` is for reading the H36M and SURREAL datasets (after processing to generate heatmaps).
