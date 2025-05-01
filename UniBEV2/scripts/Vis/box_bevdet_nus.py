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

from nuscenes.utils.data_classes import Box

cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

def pitch2matrix_shift_box(pitch):
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    return R_y


infos = mmcv.load('./data/nuscenes/bevdetv2-nuscenes_infos_val.pkl')
infos = infos['infos']
sample_annotation = mmcv.load('/mnt/cfs/algorithm/hao.lu/Code/BEVDet/data/nuscenes/v1.0-trainval/sample_annotation.json')
# cam_front_data = nusc.get('sample_data', cam_infos['sample_token'])

track_class = {'car', 'truck', 'trailer', 'bus', 'bicycle', 'motorcycle', 'pedestrian'}

index = 4478
CAM_index = 1
ann_infos = infos[index]['ann_infos']
cam_infos = infos[index]['cams'][cam_names[CAM_index]]
img_path = infos[index]['cams'][cam_names[CAM_index]]['data_path']
img = cv2.imread(img_path)
# infos[index]['gt_boxes']
# infos[index]['gt_names']
# infos[index]['sample_token']
# infos[index]['gt_boxes']
for ann, label in zip(ann_infos[0],ann_infos[1]): # [11:14] [11:14]
    # box information
    # ann = ann_infos[0]
    xyz = ann[0:3]
    wlh = [ann[4], ann[3], ann[5]]  # wlh 这是真正的wlh的顺序！！！！
    rot_axis = ann[6]
    vel = ann[7:9]
    # cam information
    # cam_infos['ego_pose']['rotation']
    # cam_infos['ego_pose']['translation']
    # cam_infos['calibrated_sensor']['rotation']
    # cam_infos['calibrated_sensor']['translation']
    # rot_axis = [1.0,0.0,0.0,0.0]
    box = Box(center=xyz, size=wlh, orientation=Quaternion(pitch2matrix_shift_box(rot_axis)), label=np.nan, score=np.nan, \
              velocity=(np.nan, np.nan, np.nan), name=None, token=None)
    # # 从世界坐标系->车身坐标系
    # box.translate(-np.array(cam_infos['ego2global_translation']))
    # box.rotate(Quaternion(cam_infos['ego2global_rotation']).inverse)
    # 从车身坐标系->相机坐标系
    box.translate(-np.array(cam_infos['sensor2ego_translation']))
    box.rotate(Quaternion(cam_infos['sensor2ego_rotation']).inverse)
    # 过滤掉不在不在图像传感器前面的点
    corners_3d = box.corners()
    # print('------------------------------', corners_3d)
    # 从相机坐标系->像素坐标系
    camera_intrinsic=cam_infos['cam_intrinsic']
    view = np.eye(4)
    view[:3, :3] = np.array(camera_intrinsic)
    in_front = corners_3d[2, :] > 0.1
    if all(in_front) is False:
        continue
    print(rot_axis)
    print(xyz)
    print(wlh)
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



cv2.imwrite('./scripts/Vis/nus_img.png', img)


