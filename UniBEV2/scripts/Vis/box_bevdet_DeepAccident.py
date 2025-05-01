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


def pitch2matrix_shift_box(pitch):
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    return R_y



cam_names = ['Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight', 'Camera_BackLeft', 'Camera_Back', 'Camera_BackRight']


infos = mmcv.load('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/DeepAccident_data/bevdetv2-DeepAccident_infos_val.pkl')
infos = infos['infos']

# cam_front_data = nusc.get('sample_data', cam_infos['sample_token'])
index = 4562
CAM_index = 1
ann_infos = infos[index]['ann_infos']
cam_infos = infos[index]['cams'][cam_names[CAM_index]]
cam_infos['lidar_to_camera_matrix'][0:3, 0:3]
img_path = infos[index]['cams'][cam_names[CAM_index]]['image_path']
lidar_to_ego_matrix = infos[index]['lidar_to_ego_matrix']
img = cv2.imread(img_path)
# infos[index]['gt_boxes']
# infos[index]['gt_names']

for ann, label in zip(ann_infos[0][18:20], ann_infos[1][18:20]):
    # box information
    # ann = ann_infos[0]
    xyz = ann[0:3][[1, 2, 0]]  # [[1,2,0]] # [2, 0, 1]
    xyz[1] = -1.0 * xyz[1] # y反了
    # xyz[2] = -1.0 * xyz[2] # z反了
    wlh = [ann[5], ann[3], ann[4]]  # 需要354来调整
    # wlh = ann[3:6]  # wlh
    rot_axis = ann[6]
    vel = ann[7:9]
    # cam information
    # cam_infos['ego_pose']['rotation']
    # cam_infos['ego_pose']['translation']
    # cam_infos['calibrated_sensor']['rotation']
    # cam_infos['calibrated_sensor']['translation']
    # rot_axis = [1.0,0.0,0.0,0.0]
    box = Box(center=xyz, size=wlh, orientation=Quaternion(rot_axis), label=np.nan, score=np.nan, \
              velocity=(np.nan, np.nan, np.nan), name=None, token=None)
    # # 从世界坐标系->车身坐标系
    # box.translate(-np.array(cam_infos['ego2global_translation']))
    # box.rotate(Quaternion(cam_infos['ego2global_rotation']).inverse)
    # # # 自车坐标系→lidar
    # box.translate(-np.array(lidar_to_ego_matrix[0:3, 3]))
    # box.rotate(Quaternion(matrix=np.array(lidar_to_ego_matrix[0:3, 0:3])).inverse)
    # 从lidar坐标系->相机坐标系
    # lidar_to_camera_matrix = cam_infos['lidar_to_camera_matrix'][0:3,0:3]
    lidar_to_camera_matrix = [[cam_infos['lidar_to_camera_matrix'][0, 0], 0, -cam_infos['lidar_to_camera_matrix'][0, 1]],
                              [0, 1, 0],
                              [cam_infos['lidar_to_camera_matrix'][0, 1], 0, cam_infos['lidar_to_camera_matrix'][0, 0]]]
        # np.linalg.inv([[ 0.01026021,  0.00843345,  0.9999118],
        #                        [0.99987258,  0.01231626,  0.01015593],
        #                        [-0.01222952, -0.99988859,  0.00855874]])
# np.linalg.inv([[0.01026021, 0.00843345, 0.9999118],
#                [-0.99987258, 0.01231626, 0.01015593],
#                [-0.01222952, -0.99988859, 0.00855874]])
#     array([[0.01026021, -0.99987257, -0.01222952],
#            [0.00843345, 0.01231626, -0.99988858],
#            [0.9999118, 0.01015593, 0.00855874]])
                            # [[-cam_infos['lidar_to_camera_matrix'][0, 1], cam_infos['lidar_to_camera_matrix'][0, 0], 0],
                            #   [0, 0, 1],
                            #   [cam_infos['lidar_to_camera_matrix'][0, 0], cam_infos['lidar_to_camera_matrix'][0, 1], 0]]
    # rot_cam1 = math.asin(cam_infos['lidar_to_camera_matrix'][0, 0])
    # rot_cam2 = math.acos(cam_infos['lidar_to_camera_matrix'][0, 1])
    box.rotate(Quaternion(matrix=np.array(lidar_to_camera_matrix)))
    temp = np.array(cam_infos['lidar_to_camera_matrix'][0:3, 3][[1, 2, 0]])
    temp[1] = -1.0 * temp[1]
    box.translate(temp)
    # 过滤掉不在不在图像传感器前面的点
    corners_3d = box.corners()
    # print(corners_3d)
    # 从相机坐标系->像素坐标系
    # camera_intrinsic=cam_infos['camera_intrinsic_matrix']
    camera_intrinsic = [[1142.5184053936916, 0.0, 800.0], [0.0, 1142.5184053936916, 450.0],
                         [0.0, 0.0, 1.0]]
    #  camera_intrinsic = [[560.16603057, 0.0, 800.0], [0.0, 560.16603057, 450.0],
    #                    [0.0, 0.0, 1.0]]
    #[[1142.5184053936916/2, 0.0, 800.0], [0.0, -1142.5184053936916/2, 450.0],
    #                 [0.0, 0.0, 1.0]]
    view = np.eye(4)
    view[:3, :3] = np.array(camera_intrinsic)
    in_front = corners_3d[2, :] > 0.1
    # print(box.center)
    # print(corners_3d)
    # print(rot_axis)
    # print(xyz)
    # print(wlh)
    # print('!!!!!!!!!!!!!!!')
    if all(in_front) is False:
        continue
    points = corners_3d
    points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
    points = np.dot(view, points)[:3, :]
    points /= points[2, :]
    box_img = points.astype(np.int32)
    color = (64, 128, 255)
    print(box.center)
    print(corners_3d)
    print(rot_axis)
    print(xyz)
    print(wlh)
    print(box_img)
    print('!!!!!!!!!!!!!!!')
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


cv2.imwrite('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/Vis/DA_img.png', img)


