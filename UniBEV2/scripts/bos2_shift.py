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
import numpy as np
from scipy.spatial.transform import Rotation

def rotation_matrix_to_quaternion(rot_matrix):
    r = Rotation.from_matrix(rot_matrix)
    quat = r.as_quat()
    r_new = Rotation.from_quat(quat)
    rot_matrix_new = r_new.as_matrix()
    assert np.allclose(rot_matrix, rot_matrix_new)
    return quat


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
    mat = np.dot(np.dot(R_z, R_y), R_x)
    return mat

import numpy as np

# def rotation_matrix_to_quaternion(rot_matrix):
#     trace = np.trace(rot_matrix)
#     if trace > 0:
#         S = np.sqrt(trace + 1.0) * 2
#         qw = 0.25 * S
#         qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
#         qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
#         qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
#     elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
#         S = np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
#         qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
#         qx = 0.25 * S
#         qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
#         qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
#     elif rot_matrix[1, 1] > rot_matrix[2, 2]:
#         S = np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
#         qw = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
#         qx = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
#         qy = 0.25 * S
#         qz = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
#     else:
#         S = np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
#         qw = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
#         qx = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
#         qy = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
#         qz = 0.25 * S
#     return np.array([qw, qx, qy, qz])


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

angle = math.pi * np.array([-1, -0.5, 0, 0.5, 1, 1.5, 2])
i = 0
for angle1 in angle:
    for angle2 in angle:
        for angle3 in angle:
            index = 1203 + i
            i = i+1
            ann_infos = infos[index]['ann_infos']
            cam_infos = infos[index]['cam_infos'][cam_names[CAM_index]]
            img_path = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_SHIFT/data/SHIFT/Train/' + \
                       infos[index]['cam_infos'][cam_names[CAM_index]]['filename']
            img = cv2.imread(img_path)
            for ann in ann_infos:
                # box information
                # ann = ann_infos[0]
                xyz = ann['translation']
                wlh = ann['size']
                rot_axis = ann['rotation']
                rot_axis[0] = rot_axis[0] + angle1
                rot_axis[1] = rot_axis[1] + angle2
                rot_axis[2] = rot_axis[2] + angle3
                # rot_axis[2] = rot_axis[2] - 1.57
                # rot_axis[1] = rot_axis[1] - 1.57
                # rot_axis[2] = rot_axis[2]
                # rot = [rot_axis[2], rot_axis[0], rot_axis[1]]
                # rot_axis[1] = rot_axis[1] - 1.57
                # rot_axis[2] = rot_axis[2] + 1.57
                # rot_axis[1] = rot_axis[1] - 1.57 #
                # rot_axis[2] = rot_axis[2] + 1.57
                zzz = Quaternion(xyz2Quaternion([- 1.57, 0, 0], 'xyz')) * Quaternion(xyz2Quaternion(rot_axis, 'xyz'))
                # rot_axis = [0,0,0]
                box = Box(center=xyz, size=wlh, orientation=zzz, label=np.nan, score=np.nan,
                          velocity=(np.nan, np.nan, np.nan), name=None, token=None)
                corners_3d = box.corners()
                print('------------------------------', corners_3d)
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
                for i in range(4):
                    j = (i + 1) % 4
                    # 下底面
                    cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, j], box_img[1, j]), color, thickness=1)
                    # 上底面
                    cv2.line(img, (box_img[0, i + 4], box_img[1, i + 4]), (box_img[0, j + 4], box_img[1, j + 4]), color,
                             thickness=1)
                    # 侧边线
                    cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, i + 4], box_img[1, i + 4]), color, thickness=1)
                    center_bottom_forward = np.mean(box_img.T[2:4], axis=0)
                    center_bottom = np.mean(box_img.T[[2, 3, 7, 6]], axis=0)
                    cv2.line(img, (int(center_bottom[0]), int(center_bottom[1])),
                             (int(center_bottom_forward[0]), int(center_bottom_forward[1])), (164, 2, 1), 2)
            cv2.imwrite('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth/scripts/' + str(angle1) + str(angle2) + str(angle2) + 'img.png', img)

# center_bottom_forward = np.mean(box_img.T[2:4], axis=0)
# center_bottom = np.mean(box_img.T[[2, 3, 7, 6]], axis=0)
# cv2.line(img, (int(center_bottom[0]), int(center_bottom[1])),(int(center_bottom_forward[0]), int(center_bottom_forward[1])), (164, 2, 1), 2)


