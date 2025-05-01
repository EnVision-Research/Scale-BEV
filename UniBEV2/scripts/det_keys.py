import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import math
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


# def print_dict_keys(dict_obj, level=0, separator='  ', sample_index=0):
#     if isinstance(dict_obj, dict):
#         for key in dict_obj.keys():
#             print(separator*level + key)
#             if isinstance(dict_obj[key], dict):
#                 print_dict_keys(dict_obj[key], level+1, separator, sample_index)
#             elif isinstance(dict_obj[key], list):
#                 print_dict_keys(dict_obj[key][sample_index], level+1, separator, sample_index)
#     elif isinstance(dict_obj, list):
#         for i, sample in enumerate(dict_obj):
#             print(separator*level + f"sample{i+1}")
#             if i == sample_index:
#                 print_dict_keys(sample, level+1, separator, sample_index)


def print_dict_keys(dict_obj, level=0, separator='  '):
    if isinstance(dict_obj, dict):
        for key in dict_obj.keys():
            print(separator*level + key)
            if isinstance(dict_obj[key], dict):
                print_dict_keys(dict_obj[key], level+1, separator)
            elif isinstance(dict_obj[key], list):
                if isinstance(dict_obj[key][0], dict):
                    print('*******' )
                    print_dict_keys(dict_obj[key][0], level+1, separator)
                else:
                    for item in dict_obj[key]:
                        print_dict_keys(item, level+1, separator)
    elif isinstance(dict_obj, list):
        for sample in dict_obj:
            print(separator*level + "sample")
            print_dict_keys(sample, level+1, separator)


def print_dict_keys(d, indent=0, separator="|"):
    for key in d.keys():
        if isinstance(d[key], dict):
            print(f"{separator*indent}{key}")
            print_dict_keys(d[key], indent=indent+1, separator=separator)
        elif isinstance(d[key], list) and d[key]:
            print(f"{separator*indent}{key}")
            if isinstance(d[key][0], dict):
                print('********************')
                print_dict_keys(d[key][0], indent=indent+1, separator=separator)
        else:
            print(f"{separator*indent}{key}")





shift_3d_path_train = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth/data/SHIFT/shift_infos_train_s.pkl'
info = mmcv.load(shift_3d_path_train)
print_dict_keys(info)

nus_3d_path_train = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth/data/nuScenes/nuscenes_infos_train.pkl'
info = mmcv.load(nus_3d_path_train)
print_dict_keys(info[0])
# sample_token
# timestamp
# scene_token
# cam_infos
# |CAM_FRONT
# ||sample_token
# ||ego_pose
# |||token
# |||timestamp
# |||rotation
# |||translation
# ||timestamp
# ||is_key_frame
# ||height
# ||width
# ||filename
# ||calibrated_sensor
# |||token
# |||sensor_token
# |||translation
# |||rotation
# |||camera_intrinsic
# |CAM_FRONT_RIGHT
# ||sample_token
# ||ego_pose
# |||token
# |||timestamp
# |||rotation
# |||translation
# ||timestamp
# ||is_key_frame
# ||height
# ||width
# ||filename
# ||calibrated_sensor
# |||token
# |||sensor_token
# |||translation
# |||rotation
# |||camera_intrinsic
# |CAM_BACK_RIGHT
# ||sample_token
# ||ego_pose
# |||token
# |||timestamp
# |||rotation
# |||translation
# ||timestamp
# ||is_key_frame
# ||height
# ||width
# ||filename
# ||calibrated_sensor
# |||token
# |||sensor_token
# |||translation
# |||rotation
# |||camera_intrinsic
# |CAM_BACK
# ||sample_token
# ||ego_pose
# |||token
# |||timestamp
# |||rotation
# |||translation
# ||timestamp
# ||is_key_frame
# ||height
# ||width
# ||filename
# ||calibrated_sensor
# |||token
# |||sensor_token
# |||translation
# |||rotation
# |||camera_intrinsic
# |CAM_BACK_LEFT
# ||sample_token
# ||ego_pose
# |||token
# |||timestamp
# |||rotation
# |||translation
# ||timestamp
# ||is_key_frame
# ||height
# ||width
# ||filename
# ||calibrated_sensor
# |||token
# |||sensor_token
# |||translation
# |||rotation
# |||camera_intrinsic
# |CAM_FRONT_LEFT
# ||sample_token
# ||ego_pose
# |||token
# |||timestamp
# |||rotation
# |||translation
# ||timestamp
# ||is_key_frame
# ||height
# ||width
# ||filename
# ||calibrated_sensor
# |||token
# |||sensor_token
# |||translation
# |||rotation
# |||camera_intrinsic
# lidar_infos
# |LIDAR_TOP
# ||sample_token
# ||ego_pose
# |||token
# |||timestamp
# |||rotation
# |||translation
# ||timestamp
# ||filename
# ||calibrated_sensor
# |||token
# |||sensor_token
# |||translation
# |||rotation
# |||camera_intrinsic
# cam_sweeps
# lidar_sweeps
# ann_infos
# ********************
# |token
# |sample_token
# |instance_token
# |visibility_token
# |attribute_tokens
# |translation
# |size
# |rotation
# |prev
# |next
# |num_lidar_pts
# |num_radar_pts
# |category_name
# |velocity



# ####
# sample
#   sample_token
#   timestamp
#   scene_token
#     weather_coarse
#     timeofday_coarse
#     weather_fine
#     timeofday_fine
#     view
#     town
#     sun_altitude_angle
#     cloudiness
#     precipitation
#     precipitation_deposits
#     wind_intensity
#     sun_azimuth_angle
#     fog_density
#     fog_distance
#     wetness
#     fog_falloff
#   cam_infos
#     left_45
#       sample_token
#       ego_pose
#         token
#         timestamp
#         rotation
#         translation
#       timestamp
#       is_key_frame
#       height
#       width
#       filename
#       calibrated_sensor
#         token
#         sensor_token
#         translation
#         rotation
#         camera_intrinsic
#           sample
#           sample
#           sample
#           sample
#           sample
#           sample
#           sample
#           sample
#           sample
#   lidar_infos
#     LIDAR_TOP
#       sample_token
#       ego_pose
#         token
#         timestamp
#         rotation
#         translation
#       timestamp
#       filename
#       calibrated_sensor
#         token
#         sensor_token
#         translation
#         rotation
#         camera_intrinsic


shift_3d_path_train = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth/data/nuScenes/bevdetv2-nuscenes_infos_train.pkl'
info = mmcv.load(shift_3d_path_train)
print_dict_keys(info)

# infos
# ********************
# |lidar_path
# |token
# |sweeps
# |cams
# ||CAM_FRONT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_FRONT_RIGHT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_FRONT_LEFT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_BACK
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_BACK_LEFT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_BACK_RIGHT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# |lidar2ego_translation
# |lidar2ego_rotation
# |ego2global_translation
# |ego2global_rotation
# |timestamp
# |gt_boxes
# |gt_names
# |gt_velocity
# |num_lidar_pts
# |num_radar_pts
# |valid_flag
# |c
# |scene_token
# metadata
# |version


lyft_3d_path_train = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/lyft/lyft_infos_ann_val.pkl'
info = mmcv.load(lyft_3d_path_train)
print_dict_keys(info)





lyft_3d_path_train = '/mnt/cfs/algorithm/hao.lu/Code/BEVDet/data/lyft/lyft_infos_train.pkl'
info = mmcv.load(lyft_3d_path_train)
print_dict_keys(info)
#
#
# infos
# ********************
# |lidar_path
# |token
# |sweeps
# ********************
# ||data_path
# ||type
# ||sample_data_token
# ||sensor2ego_translation
# ||sensor2ego_rotation
# ||ego2global_translation
# ||ego2global_rotation
# ||timestamp
# ||sensor2lidar_rotation
# ||sensor2lidar_translation
# |cams
# ||CAM_FRONT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_FRONT_RIGHT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_FRONT_LEFT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_BACK
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_BACK_LEFT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# ||CAM_BACK_RIGHT
# |||data_path
# |||type
# |||sample_data_token
# |||sensor2ego_translation
# |||sensor2ego_rotation
# |||ego2global_translation
# |||ego2global_rotation
# |||timestamp
# |||sensor2lidar_rotation
# |||sensor2lidar_translation
# |||cam_intrinsic
# |lidar2ego_translation
# |lidar2ego_rotation
# |ego2global_translation
# |ego2global_rotation
# |timestamp
# |gt_boxes
# |gt_names
# |num_lidar_pts
# |num_radar_pts
# metadata
# |version


DA_3d_path_train = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/DeepAccident_data/carla_infos_val.pkl'
info = mmcv.load(DA_3d_path_train)
print_dict_keys(info)

#
# infos
# ********************
# |scenario_type
# |vehicle_name
# |scene_name
# |lidar_prefix
# |lidar_path
# |bev_path
# |timestamp
# |scenario_length
# |cams
# ||Camera_FrontLeft
# |||image_path
# |||lidar_to_camera_matrix
# |||camera_intrinsic_matrix
# |||timestamp
# ||Camera_Front
# |||image_path
# |||lidar_to_camera_matrix
# |||camera_intrinsic_matrix
# |||timestamp
# ||Camera_FrontRight
# |||image_path
# |||lidar_to_camera_matrix
# |||camera_intrinsic_matrix
# |||timestamp
# ||Camera_BackLeft
# |||image_path
# |||lidar_to_camera_matrix
# |||camera_intrinsic_matrix
# |||timestamp
# ||Camera_Back
# |||image_path
# |||lidar_to_camera_matrix
# |||camera_intrinsic_matrix
# |||timestamp
# ||Camera_BackRight
# |||image_path
# |||lidar_to_camera_matrix
# |||camera_intrinsic_matrix
# |||timestamp
# |lidar_to_ego_matrix
# |ego_to_world_matrix
# |vehicle_speed_x
# |vehicle_speed_y
# |gt_names
# |gt_boxes
# |gt_velocity
# |vehicle_id
# |num_lidar_pts
# |camera_visibility
# metadata
# |version

