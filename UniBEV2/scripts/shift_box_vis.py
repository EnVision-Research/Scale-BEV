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
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import math

cam_names = ['left_45', 'left_stereo', 'left_45', 'left_90', 'right_45', 'right_90']
CAM_index = 0
infos = mmcv.load('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth/data/SHIFT/shift_infos_train_s.pkl')
# cam_front_data = nusc.get('sample_data', cam_infos['sample_token'])
track_class = {'car', 'truck', 'trailer', 'bus', 'bicycle', 'motorcycle', 'pedestrian'}


def rot2matrix_shift_box(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    mat = np.dot(np.dot(R_x, R_z), R_y)
    return mat

def xyz2Quaternion(xyz, euler="xyz"):
    rot = np.array(xyz)
    rot_matrix = rot2matrix_shift_box(rot[0],rot[1],rot[2])
    return rot_matrix


 # cam information
    # cam_infos['ego_pose']['rotation']
    # cam_infos['ego_pose']['translation']
    # cam_infos['calibrated_sensor']['rotation']
    # cam_infos['calibrated_sensor']['translation']

    # rot_axis = [90.0, 0.0, 1.4]
    # rot_axis = ann['rotation']

    # camera_intrinsic=cam_infos['calibrated_sensor']['camera_intrinsic']
    # camera_intrinsic = np.array(camera_intrinsic)
    # img2 = box.render_cv2(img, camera_intrinsic)


    # # 从世界坐标系->车身坐标系
    # box.translate(-np.array(cam_infos['ego_pose']['translation']))
    # box.rotate(Quaternion(cam_infos['ego_pose']['rotation']).inverse)
    #
    # 从车身坐标系->相机坐标系
    # box.translate(-np.array(cam_infos['calibrated_sensor']['translation']))
    # box.rotate(Quaternion(cam_infos['calibrated_sensor']['rotation']).inverse)

    # 过滤掉不在不在图像传感器前面的点

index = 2422
ann_infos = infos[index]['ann_infos']
cam_infos = infos[index]['cam_infos'][cam_names[CAM_index]]
img_path = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_SHIFT/data/SHIFT/Train/' + infos[index]['cam_infos'][cam_names[CAM_index]]['filename']
img = cv2.imread(img_path)
for ann in ann_infos:
    # box information
    # ann = ann_infos[2]
    xyz = ann['translation']
    height, width, length = ann['size']
    rot_axis = ann['rotation']
    wlh = [height, length, width] # [height, length, width]
    # rot_axis = [0,0,0]
    # box = Box(center=xyz, size=wlh, orientation=Quaternion(matrix=xyz2Quaternion(rot_axis, 'xyz')), label=np.nan, score=np.nan,
    #           velocity=(np.nan, np.nan, np.nan), name=None, token=None)
    # rot_axis = [1.0,0.0,0.0,0.0]
    print('****************')
    print(rot_axis[1])
    box = Box(center=xyz, size=wlh, orientation=Quaternion(rot_axis[1]), label=np.nan, score=np.nan, \
              velocity=(np.nan, np.nan, np.nan), name=None, token=None)
    print(box)
    box.rotate(Quaternion(-0.44))
    print(box.orientation.rotation_matrix)
    corners_3d = box.corners()
    # print('------------------------------', corners_3d)
    # 从相机坐标系->像素坐标系
    camera_intrinsic=cam_infos['calibrated_sensor']['camera_intrinsic']
    view = np.eye(4)
    view[:3, :3] = np.array(camera_intrinsic)
    in_front = corners_3d[2, :] > 0.1
    # if all(in_front) is False:
    #     continue
    points = corners_3d
    points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
    points = np.dot(view, points)[:3, :]
    points /= points[2, :]
    box_img = points.astype(np.int32)
    color = (64, 128, 255)
    # for i in range(4):
    #     j = (i + 1) % 4
    #     # 下底面
    #     cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, j], box_img[1, j]), color, thickness=1)
    #     # 上底面
    #     cv2.line(img, (box_img[0, i + 4], box_img[1, i + 4]), (box_img[0, j + 4], box_img[1, j + 4]), color,
    #              thickness=1)
    #     # 侧边线
    #     cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, i + 4], box_img[1, i + 4]), color, thickness=1)
    #     center_bottom_forward = np.mean(box_img.T[2:4], axis=0)
    #     center_bottom = np.mean(box_img.T[[2, 3, 7, 6]], axis=0)
    #     cv2.line(img, (int(center_bottom[0]), int(center_bottom[1])),
    #              (int(center_bottom_forward[0]), int(center_bottom_forward[1])), (164, 2, 1), 2)


cv2.imwrite('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/img1.png', img)

    # center_bottom_forward = np.mean(box_img.T[2:4], axis=0)
    # center_bottom = np.mean(box_img.T[[2, 3, 7, 6]], axis=0)
    # cv2.line(img, (int(center_bottom[0]), int(center_bottom[1])),(int(center_bottom_forward[0]), int(center_bottom_forward[1])), (164, 2, 1), 2)


