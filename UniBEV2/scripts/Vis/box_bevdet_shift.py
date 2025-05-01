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

infos = mmcv.load('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/PKL/bevdetv2_shift_infos_val_v1.pkl')
infos = infos['infos']

# cam_front_data = nusc.get('sample_data', cam_infos['sample_token'])

track_class = {'car', 'truck', 'trailer', 'bus', 'bicycle', 'motorcycle', 'pedestrian'}

index = 24452
CAM_index = 0
ann_infos = infos[index]['ann_infos']
ann_infos_v1 = infos[index]['ann_infos_v1']
cam_infos = infos[index]['cams'][cam_names[CAM_index]]
# img_path = infos[index]['cams'][cam_names[CAM_index]]['data_path']
img_path = infos[index]['cams'][cam_names[CAM_index]]['data_path']
# img = cv2.imread('/mnt/cfs/algorithm/hao.lu/Data/SHIFT_Pro/'+img_path)
img = cv2.imread(img_path)

iii=0
for ann, label in zip(ann_infos[0],ann_infos[1]):
    # box information
    # ann = ann_infos[0]
    xyz = ann[0:3]
    wlh = [ann[4],ann[5],ann[3]]
    # wlh = ann[3:6]  # wlh
    rot_axis = ann[6]
    rot_axis = rot_axis
    vel = ann[7:9]
    # cam information
    # cam_infos['ego_pose']['rotation']
    # cam_infos['ego_pose']['translation']
    # cam_infos['calibrated_sensor']['rotation']
    # cam_infos['calibrated_sensor']['translation']
    # rot_axis = [1.0,0.0,0.0,0.0]
    # -rot_axis-1.57
    # box = Box(center=xyz, size=wlh, orientation=Quaternion(matrix=pitch2matrix_shift_box( -rot_axis-1.57)), label=np.nan, score=np.nan, \
    #           velocity=(np.nan, np.nan, np.nan), name=None, token=None)
    box = Box(center=xyz, size=wlh, orientation=Quaternion(matrix=pitch2matrix_shift_box(rot_axis)), label=np.nan,
              score=np.nan, \
              velocity=(np.nan, np.nan, np.nan), name=None, token=None)
    # 世界坐标系→相机坐标系
    box.rotate(Quaternion(matrix=pitch2matrix_shift_box(math.radians(math.radians(ann_infos_v1[iii]['ego_rotation'])))))
    box.translate(np.array(ann_infos_v1[iii]['ego_translation']))
    box.translate(-np.array(ann_infos_v1[iii]['word2cam_translate']))
    box.rotate(Quaternion(matrix=pitch2matrix_shift_box(math.radians(ann_infos_v1[iii]['word2cam_rotate']))).inverse)
    if xyz[0]>50 or xyz[0]<-50:
        iii = iii + 1
        continue
    if xyz[2]>50 or xyz[2]<-50:
        iii = iii + 1
        continue
    if xyz[0]>xyz[2] or xyz[0]<-xyz[2]:
        iii = iii + 1
        continue
    # if rot_axis>3 or rot_axis<-3:
    #     continue
    # # 从世界坐标系->车身坐标系
    # box.translate(-np.array(cam_infos['ego2global_translation']))
    # box.rotate(Quaternion(cam_infos['ego2global_rotation']).inverse)
    # 从车身坐标系->相机坐标系
    # box.translate(-np.array(cam_infos['sensor2ego_translation']))
    # box.rotate(Quaternion(cam_infos['sensor2ego_rotation']).inverse)
    # 过滤掉不在不在图像传感器前面的点
    corners_3d = box.corners()
    # print('------------------------------', corners_3d)
    # 从相机坐标系->像素坐标系
    camera_intrinsic=cam_infos['calibrated_sensor']['camera_intrinsic']
    view = np.eye(4)
    view[:3, :3] = np.array(camera_intrinsic)
    in_front = corners_3d[2, :] > 0.1
    if all(in_front) is False:
        iii = iii + 1
        continue
    # if ann_infos_v1[iii]['cam_name'] != 'front':
    #     iii = iii + 1
    #     continue
    ann_infos_v1[iii]
    # if ann_infos_v1[iii]['category'] != 'car':
    #     iii = iii + 1
    #     continue
    # if iii >2:
    #     break
    # print('!!!')
    #print(xyz)
    print(rot_axis)
    #print(wlh)
    #print(ann_infos_v1[iii])
    iii = iii + 1
    points = corners_3d
    points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
    points = np.dot(view, points)[:3, :]
    points /= points[2, :]
    box_img = points.astype(np.int32)
    color = (64, 128, 255)
    # b = list(set(ann['category_name'].split('.')).intersection(track_class))
    # if len(b) == 0:
    #     continue
    # else:
    for i in range(4):
        j = (i + 1) % 4
        # 下底面
        cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, j], box_img[1, j]), color, thickness=1)
        # 上底面
        cv2.line(img, (box_img[0, i + 4], box_img[1, i + 4]), (box_img[0, j + 4], box_img[1, j + 4]), color,
                 thickness=1)
        # 侧边线
        cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, i + 4], box_img[1, i + 4]), color, thickness=1)
        # # 蓝线定义
        # center_bottom_forward = np.mean(box_img.T[2:4], axis=0)
        # center_bottom = np.mean(box_img.T[[2, 3, 7, 6]], axis=0)
        # cv2.line(img, (int(center_bottom[0]), int(center_bottom[1])),
        #          (int(center_bottom_forward[0]), int(center_bottom_forward[1])), (164, 2, 1), 2)



cv2.imwrite('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/Vis/shift_img.png', img)


