# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import mmcv


cam_names = ['Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight', 'Camera_BackLeft', 'Camera_Back', 'Camera_BackRight']

map_name_from_general_to_detection = {
    'car':'car',
    'truck':'truck',
    'van':'truck',
    'bus':'bus',
    'emergency_vehicle':'ignore',
    'other_vehicle':'car',
    'motorcycle':'motorcycle',
    'bicycle':'bicycle',
    'cyclist':'bicycle',
    'pedestrian':'pedestrian',
    'animal':'ignore',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
# classes = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
classes = [
    'car', 'truck', 'van',
    'motorcycle', 'bicycle', 'pedestrian'
] # label 的顺序0-5

general_to_coarse = {
    'pedestrian': 'pedestrian',
    'motorcycle': 'cycle',
    'bicycle': 'cycle',
    'car': 'car',
    'bus': 'car',
    'van': 'car',
    'trailer': 'car',
    'truck': 'car',
    'construction_vehicle': 'car',
    'barrier': 'ignore',
    'traffic_cone': 'ignore',
}

classes_3 = [
    'car', 'cycle', 'pedestrian', 'ignore'
]


def get_gt_DeepAccident2nus(info):
    gt_boxes = list()
    gt_labels = list()
    for id in range(len(info['gt_boxes'])):
        if info['camera_visibility'][id]==1:
            temp_box = info['gt_boxes'][id]
            temp_box[1] = -1.0 * temp_box[1]
            temp_box[2] = temp_box[2] + 1.9
            gt_boxes.append(np.concatenate([info['gt_boxes'][id], info['gt_velocity'][id]]))
            gt_labels.append(
                classes_3.index(
                    general_to_coarse[map_name_from_general_to_detection[info['gt_names'][id]]]))
    # print(info['gt_names'])
    return gt_boxes, gt_labels

def add_ann_adj_info(dataset, save_path, dataroot='./data/DeepAccident/'):
    for id in range(len(dataset['infos'])):
        if id % 10 == 0:
            print('%d/%d' % (id, len(dataset['infos'])))
        info = dataset['infos'][id]
        # get sweep adjacent frame info
        dataset['infos'][id]['ann_infos'] = get_gt_DeepAccident2nus(info)
        dataset['infos'][id]['scene_token'] = info['scenario_type'] + info['vehicle_name'] +  info['scene_name']
        # get sensor2ego_translation
        for cam_name in cam_names:
            cam_infos = dataset['infos'][id]['cams'][cam_name]
            lidar2cam_rt = cam_infos['lidar_to_camera_matrix']
            ego2lidar_rt = np.linalg.inv(dataset['infos'][id]['lidar_to_ego_matrix'])
            ego2cam = lidar2cam_rt @ ego2lidar_rt
            ego2cam_rt = ego2cam.T
            ego2cam_rt = ego2cam_rt[[1, 2, 0], 0:3]
            ego2cam_rt[1, :] = -ego2cam_rt[1, :]
            ego2cam_rt[0, :] = -ego2cam_rt[0, :]
            lidar_to_camera_matrix = ego2cam_rt
            temp = np.array(cam_infos['lidar_to_camera_matrix'][0:3, 3][[1, 2, 0]])
            temp[1] = -1.0 * temp[1]
            view = np.eye(4)
            view[:3, :3] = np.array(lidar_to_camera_matrix)
            view[:3, -1] = temp
            ego2img = np.linalg.inv(view)
            ego2im_Q = Quaternion(matrix=np.array(ego2img[0:3, 0:3]))
            sensor2ego_rotation = [ego2im_Q.w.astype(float), ego2im_Q.x.astype(float), ego2im_Q.y.astype(float),
                                   ego2im_Q.z.astype(float)]
            dataset['infos'][id]['cams'][cam_name]['sensor2ego_rotation'] = sensor2ego_rotation
            dataset['infos'][id]['cams'][cam_name]['sensor2ego_translation'] = (ego2img[0:3, -1] + np.array([0.0,0.0,1.9])).tolist()
    with open(save_path,
              'wb') as fid:
        pickle.dump(dataset, fid)

# classes = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]

if __name__ == '__main__':
    DeepAccident_pkl_path = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/DeepAccident_data/carla_infos_train.pkl'
    DeepAccident_data = mmcv.load(DeepAccident_pkl_path)
    add_ann_adj_info(dataset=DeepAccident_data, save_path='/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/DeepAccident_data/DA-DeepAccident_infos_train.pkl')

    DeepAccident_pkl_path = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/DeepAccident_data/carla_infos_val.pkl'
    DeepAccident_data = mmcv.load(DeepAccident_pkl_path)
    add_ann_adj_info(dataset=DeepAccident_data, save_path='/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/DeepAccident_data/DA-DeepAccident_infos_val.pkl')
#
