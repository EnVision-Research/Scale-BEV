import os.path
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import mmcv
import cv2
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R

from nuscenes.utils.data_classes import Box

cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


infos = mmcv.load('/mnt/cfs/algorithm/hao.lu/Code/BEVDet/data/lyft/lyft_infos_ann_val.pkl')
infos = infos['infos']

# cam_front_data = nusc.get('sample_data', cam_infos['sample_token'])

track_class = {'car', 'truck', 'trailer', 'bus', 'bicycle', 'motorcycle', 'pedestrian'}

index = 2422
CAM_index = 1
ann_infos = infos[index]['ann_infos']
cam_infos = infos[index]['cams'][cam_names[CAM_index]]
INFOS_CAM_INTRINSIC_ = cam_infos['cam_intrinsic']
img_path = infos[index]['cams'][cam_names[CAM_index]]['data_path']
img = cv2.imread(img_path)


Q_temp = Quaternion(cam_infos['sensor2ego_rotation'])
Q_inv = Q_temp.inverse
Rq = [Q_inv.x.astype(float), Q_inv.y.astype(float), Q_inv.z.astype(float),
      Q_inv.w.astype(float)]
Rm = R.from_quat(Rq)
rotation_matrix = Rm.as_matrix()
print('rotation:\n', rotation_matrix)
