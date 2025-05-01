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
import math

from nuscenes.utils.data_classes import Box

cam_names = [
            'front', 'left_stereo', 'left_45', 'left_90',
            'right_45', 'right_90'
        ]

def pitch2matrix_shift_box(pitch):
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    return R_y

def xyz2Quaternion(xyz, euler="xyz"):
    rot = np.array(xyz)
    rot_matrix = rot2matrix_shift_box(rot[0],rot[1],rot[2])
    return rot_matrix


def find_token(ann_infos_v1,token_name):
    index_list = []
    for index, ann_v1 in enumerate(ann_infos_v1):
        if ann_v1['instance_token']==token_name:
            index_list.append(index)
    return index_list




#
#
# for ann_v1 in ann_infos_v1:
#     ann_v1['instance_token']
#
#     find_token(ann_infos_v1, '904')
#
#
# # find_token(ann_infos_v1, '901')
# # [2, 13, 26]

infos = mmcv.load('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/PKL/bevdetv2_shift_infos_val_v1.pkl')
infos = infos['infos']
# cam_front_data = nusc.get('sample_data', cam_infos['sample_token'])
track_class = {'car', 'truck', 'trailer', 'bus', 'bicycle', 'motorcycle', 'pedestrian'}
index = 8746
CAM_index = 0
ann_infos = infos[index]['ann_infos']
ann_infos_v1 = infos[index]['ann_infos_v1']
for ann in ann_infos_v1:
    print(ann['instance_token'])
find_token(ann_infos_v1, '4601')

ann_infos[0][1]
ann_infos[0][12]
ann_infos[0][25]
ann_infos[0][58]


ann_infos_v1[0]
ann_infos_v1[39]
ann_infos_v1[42]
ann_infos_v1[58]


box1 = Box(center=ann_infos_v1[0]['translation'], size=ann_infos_v1[0]['size'], orientation=Quaternion(ann_infos_v1[0]['orientation'][1]), label=np.nan, score=np.nan, \
              velocity=(np.nan, np.nan, np.nan), name=None, token=None)
box2 = Box(center=ann_infos_v1[39]['translation'], size=ann_infos_v1[39]['size'], orientation=Quaternion(ann_infos_v1[39]['orientation'][1]), label=np.nan, score=np.nan, \
              velocity=(np.nan, np.nan, np.nan), name=None, token=None)
# 相机坐标系→世界坐标系
# print(cam_name)
box1.rotate(Quaternion(matrix=pitch2matrix_shift_box(math.radians(ann_infos_v1[0]['word2cam_rotate']))))
box2.rotate(Quaternion(matrix=pitch2matrix_shift_box(math.radians(ann_infos_v1[39]['word2cam_rotate']))))
a1 = ann_infos_v1[0]['word2cam_translate']
a2 = ann_infos_v1[39]['word2cam_translate']
box1.translate(np.array([a1[1],a1[2],a1[0]]))
box2.translate(np.array([a2[1],a2[2],a2[0]]))

## 世界坐标系→相机坐标系
box1.translate(-np.array([a1[1],a1[2],a1[0]]))
box2.translate(-np.array([a2[1],a2[2],a2[0]]))
box1.rotate(Quaternion(matrix=pitch2matrix_shift_box(math.radians(ann_infos_v1[0]['word2cam_rotate']))).inverse)
box2.rotate(Quaternion(matrix=pitch2matrix_shift_box(math.radians(ann_infos_v1[39]['word2cam_rotate']))).inverse)
box1
box2

# ann_infos_v1[2]['orientation'][1] + math.radians(ann_infos_v1[1]['word2cam_rotate']) - math.radians(ann_infos_v1[1]['ego_rotation'])
# ann_infos_v1[25]['orientation'][1] + math.radians(ann_infos_v1[25]['word2cam_rotate']) - math.radians(ann_infos_v1[25]['ego_rotation'])
#
#
# box1.rotate(Quaternion(matrix=pitch2matrix_shift_box(math.radians(ann_infos_v1[2]['word2cam_rotate']+0.0000000001))).inverse)
# # 相机坐标系→世界坐标系
# box2.rotate(Quaternion(matrix=pitch2matrix_shift_box(math.radians(-ann_infos_v1[26]['word2cam_rotate']+0.0000000001))).inverse)
# box1
# box2
#
# box1.center
# # 世界坐标系→自车坐标系
# box1.rotate(Quaternion(ann_infos_v1[2]['ego_rotation'] + 0.0000000001))
# box1.translate(np.array(ann_infos_v1[2]['ego_translation']))
# # 世界坐标系→自车坐标系
# box2.rotate(Quaternion(ann_infos_v1[26]['ego_rotation'] + 0.0000000001))
# box2.translate(np.array(ann_infos_v1[26]['ego_translation']))
# box1
# box2

