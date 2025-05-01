import pickle
import os
import numpy as np
from pyquaternion.quaternion import Quaternion
from copy import deepcopy
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
import copy
from PIL import Image

# data = pickle.load(open('/remote-home/hao.lu/Code/mmdetection3d/waymo_infos_val.pkl','rb'))

# with open('frame.pkl', 'wb') as file:
#     pickle.dump(data[0], file)

all_data = pickle.load(open('/mnt/cfs/algorithm/hao.lu/Code/waymo_pre/mmdetection3d/data/waymo/kitti_format/waymo_infos_val.pkl','rb'))
all_calib = np.load('/mnt/cfs/algorithm/hao.lu/Code/waymo_pre/mmdetection3d/calib_val.npy')

map_name_from_general_to_detection = {
    'bicycle': 'bicycle',
    'Cyclist': 'bicycle',
    'bus': 'car',
    'Car': 'car',
    'emergency_vehicle': 'car',
    'motorcycle': 'bicycle',
    'other_vehicle': 'car',
    'Pedestrian': 'pedestrian',
    'truck': 'car',
    'animal': 'animal'
}
# map_name_from_general_to_detection = {
#     'bicycle': 'bicycle',
#     'Cyclist': 'bicycle',
#     'bus': 'bus',
#     'Car': 'car',
#     'emergency_vehicle': 'emergency_vehicle',
#     'motorcycle': 'motorcycle',
#     'other_vehicle': 'other_vehicle',
#     'Pedestrian': 'pedestrian',
#     'truck': 'truck',
#     'animal': 'animal'
# }

classes = [
    'car', 'bicycle', 'pedestrian'#, 'truck', 'bus', 'emergency_vehicle', 'other_vehicle', 'motorcycle', 'animal'
]

def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def ego2img(points_ego, camrera_info, camera2img, geo_flag=False, resize=None):
    points_lidar_homogeneous = \
        np.concatenate([points_ego,
                        np.ones((points_ego.shape[0], 1),
                                dtype=points_ego.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    # temp = camrera_info['sensor2ego_rotation']
    # camera2lidar[:3, :3] = Rotation.from_quat([temp[1], temp[2], temp[3], temp[0]]).as_matrix()
    camera2lidar[:3, :3] = Quaternion(camrera_info['sensor2ego_rotation']).rotation_matrix
    camera2lidar[:3, 3] = camrera_info['sensor2ego_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 1.0, valid)
    points_img = points_camera @ camera2img.T
    depth = copy.deepcopy(points_img[:, 2:3])
    if geo_flag:
        GeoMetric = self.get_GeoMetric(points_img)
        GeoMetric = GeoMetric * resize
    else:
        GeoMetric = None
    points_img = points_img / points_img[:, 2:3]
    points_img = points_img[:, :2]
    return points_img, valid, depth, GeoMetric

new_pkl_file = {}
all_frame_new_data = []
for idx in range(len(all_data)):
    data = all_data[idx]
    calib = all_calib[idx]

    single_frame_new_data = {}

    lidar_path = os.path.join('./data/waymo/kitti_format/',data['point_cloud']['velodyne_path'])
    token = ''
    sweeps = ''
    cams = {}
    # views = [
    #     'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
    #     'CAM_BACK_RIGHT']
    views = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT']


    image_path = data['image']['image_path']
    for i, view in enumerate(views):
        image_path = image_path.replace(image_path[15], str(i), 1)
        data_path = os.path.join('./data/waymo/kitti_format/',image_path)
        token = ''
        cam_type = view
        sample_data_token = ''
        cam_2_ego = calib[i]
        # sensor2ego_rotation = cam_2_ego[:3, :3]

        sensor2ego_rotation = np.array(list(Quaternion(matrix=cam_2_ego[:3, :3])))
        sensor2ego_translation = cam_2_ego[:3, 3]
        # ego2global_rotation = data['pose'][:3,:3]
        ego2global_rotation = np.array(list(Quaternion(matrix=data['pose'][:3,:3])))
        ego2global_translation = data['pose'][:3, 3]
        timestamp = data['timestamp']
        # sensor2lidar_rotation = sensor2ego_rotation
        sensor2lidar_rotation = cam_2_ego[:3, :3] #TODO 不需要四元数的形式
        sensor2lidar_translation = sensor2ego_translation
        cam_intrinsic = data['calib']['P{}'.format(i)][:3,:3]

        single_cam = {
                        'data_path': data_path,
                        'type': cam_type,
                        'sample_data_token': sample_data_token,
                        'sensor2ego_translation': sensor2ego_translation,
                        'sensor2ego_rotation': sensor2ego_rotation,
                        'ego2global_translation': ego2global_translation,
                        'ego2global_rotation': ego2global_rotation,
                        'timestamp': timestamp,
                        'sensor2lidar_rotation': sensor2lidar_rotation,
                        'sensor2lidar_translation': sensor2lidar_translation,
                        'cam_intrinsic': cam_intrinsic,
        }
        cams[view] = single_cam
    img = Image.open(cams['CAM_FRONT']['data_path'])
    width, height = img.size





    #这里先设定为1，以及0，由于在pkl中没有保存这个信息
    lidar2ego_translation = np.array([0,0,0])
    lidar2ego_rotation = np.array([1.0, 0.0, 0.0, 0.0])

    ego2global_translation = data['pose'][:3, 3]
    # ego2global_rotation = data['pose'][:3,:3]
    ego2global_rotation = np.array(list(Quaternion(matrix=data['pose'][:3,:3])))
    timestamp = data['timestamp']


    # get the label, l,h,w 顺序需要更改
    annos = data['annos']
    dimensions = annos['dimensions']
    rotation_y = annos['rotation_y'][:,np.newaxis]
    location = annos['location']
    location = np.concatenate([location, np.ones((len(location),1))],axis = -1)
    location = np.linalg.inv(data['calib']['Tr_velo_to_cam']) @ location.T
    location = location[:3].T
    gt_boxes = np.hstack([location,dimensions,rotation_y, np.ones((len(location), 2))])


    gt_boxes[:,6] = -(gt_boxes[:,6] + np.pi/2)

    tmp = deepcopy(gt_boxes[:,4])
    gt_boxes[:,4] = gt_boxes[:,5]
    gt_boxes[:,5] = tmp
    gt_boxes[:,2] += gt_boxes[:,5]/2

    gt_names = data['annos']['name']
    gt_velocity = np.ones(len(gt_boxes)) * (-1) # no velocity in Waymo
    num_lidar_pts = data['annos']['num_points_in_gt']
    num_radar_pts = []
    valid_flag = [True] * len(gt_boxes)
    gt_labels = list()
    gt_boxes_final = list()
    z = -1
    for name in gt_names:
        z = z + 1
        if map_name_from_general_to_detection[name] not in classes:
            continue
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[name]))
        gt_boxes_final.append(gt_boxes[z])
    if len(gt_boxes_final) < 1:
        continue
    # 判断gt_boxes是否在img上
    gt_boxes_part = np.array(gt_boxes_final)[:, 0:7]
    num_boxes, _ = gt_boxes_part.shape
    valid_flag = np.zeros(num_boxes, dtype=bool)
    for i, view in enumerate(views):
        gravity_center_lidar_gt = LB(gt_boxes_part, origin=(0.5, 0.5, 0.5)).gravity_center.numpy()
        gravity_center_img, valid, depth, _ = ego2img(gravity_center_lidar_gt, cams[view],
                                                      np.array(cams[view]['cam_intrinsic']), geo_flag=False,
                                                      resize=None)
        valid_2 = check_point_in_img(gravity_center_img, height, width)
        valid_3 = np.logical_and(valid, valid_2)
        valid_flag = valid_flag + valid_3

    selected_gt_boxes = [sample for sample, select in zip(gt_boxes_final, valid_flag) if select]
    selected_gt_labels = [sample for sample, select in zip(gt_labels, valid_flag) if select]
    print(len(gt_boxes_final), len(selected_gt_boxes))
    ann_infos = selected_gt_boxes, selected_gt_labels
    # 如果不挑选，就用下面这个（这个是点云所有label）
    # ann_infos = gt_boxes_final, gt_labels
    scene_token = []

    single_frame_new_data = {
                'lidar_path' : lidar_path,
                'token' : lidar_path,
                'sweeps' : sweeps,
                'cams' : cams,
                'lidar2ego_translation' : lidar2ego_translation,
                'lidar2ego_rotation' : lidar2ego_rotation,
                'ego2global_translation' : ego2global_translation,
                'ego2global_rotation' : ego2global_rotation,
                'timestamp' : timestamp,
                'gt_boxes' : gt_boxes,
                'gt_names' : gt_names,
                'gt_velocity' : gt_velocity,
                'num_lidar_pts' : num_lidar_pts,
                'num_radar_pts' : num_radar_pts,
                'valid_flag' : valid_flag,
                'ann_infos' : ann_infos,
                'scene_token' : scene_token,
    }

    if len(gt_boxes) > 0:
        all_frame_new_data.append(single_frame_new_data)
    else:
        print('no gt_boxes')


new_pkl_file = {
                'infos' : all_frame_new_data,
                'metadata' : {'version': 'v1.0-trainval'}
}


with open('/mnt/cfs/algorithm/hao.lu/Code/waymo_pre/mmdetection3d/data/waymo/kitti_format/Uni_waymo_infos_val_v2.pkl', 'wb') as file:
    pickle.dump(new_pkl_file, file)

