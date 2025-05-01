import os.path
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import math
import mmcv
import cv2
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R
from nuscenes.utils.data_classes import Box
from PIL import Image


def pitch2matrix_shift_box(pitch):
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    return R_y


def intri_aug(img, intrin):
    import random
    import copy
    fc_w_ratio = 0.4
    fc_h_ratio = 0.4
    # 输入一个图像
    # 进行crop改变光心
    # resize改变视场角
    height, width = img.height, img.width
    crop_x_intri, crop_y_intri = int(random.randint(0, width * fc_w_ratio)), int(random.randint(0, height * fc_h_ratio))
    fx_resize, fy_resize = np.random.uniform((0.5, 1.5))
    fx_resize = 2.1
    print(fx_resize)
    print(fy_resize)
    intrin_aug = copy.deepcopy(intrin)
    intrin_aug = np.array(intrin_aug)
    intrin_aug[0, 2] = intrin_aug[0, 2] - crop_x_intri
    intrin_aug[1, 2] = intrin_aug[1, 2] - crop_y_intri
    print(intrin_aug)
    intrin_aug[0, :] = (fx_resize) * intrin_aug[0, :]
    intrin_aug[1, :] = (fy_resize) * intrin_aug[1, :]
    print(intrin_aug)
    return intrin_aug, fx_resize, fy_resize, crop_x_intri, crop_y_intri

def img_aug(img, fx_resize, fy_resize, crop_x_intri, crop_y_intri):
    print(img.width, img.height)
    img = img.crop((crop_x_intri, crop_y_intri, img.width, img.height))
    resize_dims = (int(img.width*fx_resize), int(img.height*fy_resize))
    print(resize_dims)
    img = img.resize(resize_dims)
    print(img.width, img.height)
    return img

cam_names = ['Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight', 'Camera_BackLeft', 'Camera_Back', 'Camera_BackRight']


infos = mmcv.load('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/DeepAccident_data/bevdetv4-DeepAccident_infos_val.pkl')
infos = infos['infos']

index = 1425
CAM_index = 1
ann_infos = infos[index]['ann_infos']
cam_infos = infos[index]['cams'][cam_names[CAM_index]]
cam_infos['lidar_to_camera_matrix'][0:3, 0:3]
img_path = infos[index]['cams'][cam_names[CAM_index]]['image_path']
lidar_to_ego_matrix = infos[index]['lidar_to_ego_matrix']
img = Image.open(img_path)
camera_intrinsic = [[1142.5184053936916, 0.0, 800.0],
                    [0.0, 1142.5184053936916, 450.0],
                    [0.0, 0.0, 1.0]]

camera_intrinsic, fx_resize, fy_resize, crop_x_intri, crop_y_intri = intri_aug(img, camera_intrinsic)
img = img_aug(img, fx_resize, fy_resize, crop_x_intri, crop_y_intri)
# 转换为numpy数组
img = np.array(img)
# 转换为OpenCV格式
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
zzz = 0
for ann, label in zip(ann_infos[0], ann_infos[1]):
    zzz = zzz+1
    xyz = ann[0:3]
    wlh = [ann[4], ann[3], ann[5]]  # 需要354来调整
    rot_axis = ann[6]
    vel = ann[7:9]
    box = Box(center=xyz, size=wlh, orientation=Quaternion(rot_axis), label=np.nan, score=np.nan, \
              velocity=(np.nan, np.nan, np.nan), name=None, token=None)
    box.translate(-np.array(cam_infos['sensor2ego_translation']))
    box.rotate(Quaternion(cam_infos['sensor2ego_rotation']).inverse)
    # 过滤掉不在不在图像传感器前面的点
    corners_3d = box.corners()
    view = np.eye(4)
    view[:3, :3] = np.array(camera_intrinsic)
    in_front = corners_3d[2, :] > 0.1
    if all(in_front) is False:
        continue
    points = corners_3d
    points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
    points = np.dot(view, points)[:3, :]
    points /= points[2, :]
    box_img = points.astype(np.int32)
    color = (64, 128, 255)
    # print('-------------')
    # print(zzz)
    # print(xyz)
    # print(Quaternion(cam_infos['sensor2ego_rotation']).inverse)
    # print(box.center)
    # print(wlh)
    # print(corners_3d)
    # print(rot_axis)
    # print(box_img)
    # print('!!!!!!!!!!!!!!!')
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


cv2.imwrite('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/Vis_PKL_fixed/DA_img.png', img)


