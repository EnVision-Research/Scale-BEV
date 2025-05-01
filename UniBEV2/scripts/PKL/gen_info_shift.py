import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import math
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

translation = {'front': [0.5, 0.0, 0.0],
               'left_stereo': [],
               'left_45': [],
               'left_90': [],
               'right_45': [],
               'right_90': []

}

def pitch2matrix_shift_box(pitch):
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    return R_y

map_name_from_general_to_detection = {
    'car':'car',
    'truck':'truck',
    'bus':'bus',
    'emergency_vehicle':'ignore',
    'other_vehicle':'car',
    'motorcycle':'motorcycle',
    'bicycle':'bicycle',
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

def rpy2quaternion(roll, pitch, yaw):
    # roll, pitch, yaw = 180*roll/math.pi, 180*pitch/math.pi, 180*roll/yaw.pi
    x=math.sin(pitch/2)*math.sin(yaw/2)*math.cos(roll/2)+math.cos(pitch/2)*math.cos(yaw/2)*math.sin(roll/2)
    y=math.sin(pitch/2)*math.cos(yaw/2)*math.cos(roll/2)+math.cos(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)
    z=math.cos(pitch/2)*math.sin(yaw/2)*math.cos(roll/2)-math.sin(pitch/2)*math.cos(yaw/2)*math.sin(roll/2)
    w=math.cos(pitch/2)*math.cos(yaw/2)*math.cos(roll/2)-math.sin(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)
    return [x, y, z, w]
def xyz2Quaternion(xyz, euler="xyz"):
    rot = np.array(xyz)
    rot_matrix = R.from_euler(euler, rot, degrees=False).as_matrix()
    return Quaternion(matrix=rot_matrix)
def yaw_nom(yaw):
    if yaw > math.pi:
        yaw = math.pi - 2*math.pi
    elif yaw < -math.pi:
        yaw = math.pi + 2 * math.pi
    return yaw
def yaw_lyft2nus(yaw):
    yaw = yaw_nom(yaw)
    yaw = -yaw - (math.pi/2)
    yaw = yaw_nom(yaw)
    return [yaw]

cam_names = [
            'front', 'left_stereo', 'left_45', 'left_90',
            'right_45', 'right_90'
        ]

classes = [
    'car', 'truck', 'bus',
    'motorcycle', 'bicycle', 'pedestrian'
]

def info_transfor(infos_old, max_cam_sweeps=6, max_lidar_sweeps=1):
    infos_new = list() # infos_old = infos_train
    det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/left_stereo/det_3d.json'
    infos_ls = mmcv.load(det_3d_path)
    det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/front/det_3d.json'
    infos_f = mmcv.load(det_3d_path)
    det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/left_45/det_3d.json'
    infos_l45 = mmcv.load(det_3d_path)
    det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/left_90/det_3d.json'
    infos_l90 = mmcv.load(det_3d_path)
    det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/right_45/det_3d.json'
    infos_r45 = mmcv.load(det_3d_path)
    det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/right_90/det_3d.json'
    infos_r90 = mmcv.load(det_3d_path)
    infos_all = dict()
    infos_all['front'] = infos_f
    infos_all['left_stereo'] = infos_ls
    infos_all['left_45'] = infos_l45
    infos_all['left_90'] = infos_l90
    infos_all['right_45'] = infos_r45
    infos_all['right_90'] = infos_r90

    index = 0
    for data_frames in infos_old['frames']:  # data_frames=infos_old['frames'][0]
        print(data_frames['videoName'])
        info = dict()

        info['token'] = data_frames['videoName'] + '_' + data_frames['name'][0:8]
        info['sample_token'] = data_frames['videoName'] + '_' + data_frames['name'][0:8]
        info['timestamp'] = data_frames['name'][0:8]
        info['scene_token'] = data_frames['attributes']

        lidar_names = ['LIDAR_TOP']
        cam_infos = dict()
        lidar_infos = dict()
        # 前向摄像头作为方向，左右摄像头求中心
        # ego_translation = list(map(lambda x: (x[0] + x[1])/2,
        #          zip(infos_all['left_90']['frames'][index]['extrinsics']['location'], infos_all['right_90']['frames'][index]['extrinsics']['location'])))
        ego_translation = infos_all['front']['frames'][index]['extrinsics']['location']

        ego_translation = np.array([ego_translation[1], ego_translation[2], ego_translation[0]])
        # print('ego_translation', ego_translation)
        ego_rotation = infos_all['front']['frames'][index]['extrinsics']['rotation'][2]
        # print('ego_rotation', infos_all['front']['frames'][index]['extrinsics']['rotation'])
        for cam_name in cam_names:
            sweep_cam_info = dict()
            sweep_cam_info['sample_token'] = info['sample_token']
            sweep_cam_info['ego_pose'] = {'token': info['sample_token'] + '_' + cam_name,
                                           'timestamp': data_frames['name'][0:8],
                                           'rotation': ego_rotation,
                                           'translation': ego_translation}
            sweep_cam_info['timestamp'] = info['timestamp']
            sweep_cam_info['is_key_frame'] = True
            sweep_cam_info['height'] = 1280
            sweep_cam_info['width'] = 800
            sweep_cam_info['data_path'] = 'data/SHIFT/Train/' + cam_name + '/' + data_frames['videoName'] + '/' + \
                                         infos_all[cam_name]['frames'][index]['name']

            # sweep_cam_info['sensor2ego_translation'] = ego_translation-infos_all[cam_name]['frames'][index]['extrinsics']['location']
            # sweep_cam_info['sensor2ego_rotation'] = ego_rotation/(infos_all[cam_name]['frames'][index]['extrinsics']['rotation'][2]+0.000001)

            sweep_cam_info['filename'] = cam_name + '/' + data_frames['videoName'] + '/' + infos_all[cam_name]['frames'][index]['name']
            sweep_cam_info['camera_intrinsic'] = [[640.0, 0.0, 640.0], [0.0, 640.0, 400.0], [0.0, 0.0, 1.0]]
            sweep_cam_info['cam_intrinsic'] = [[640.0, 0.0, 640.0], [0.0, 640.0, 400.0], [0.0, 0.0, 1.0]]
            sweep_cam_info['calibrated_sensor'] = {'token': info['sample_token'] + '_' + cam_name,
                                                   'sensor_token': cam_name,
                                                   'translation': infos_all[cam_name]['frames'][index]['extrinsics']['location'],
                                                   'rotation': infos_all[cam_name]['frames'][index]['extrinsics']['rotation'][2],
                                                   'camera_intrinsic': [[640.0, 0.0, 640.0], [0.0, 640.0, 400.0],
                                                                        [0.0, 0.0, 1.0]],
                                                   'cam_intrinsic': [[640.0, 0.0, 640.0], [0.0, 640.0, 400.0],
                                                                        [0.0, 0.0, 1.0]]
                                                   }

            cam_infos[cam_name] = sweep_cam_info

        for lidar_name in lidar_names:
            sweep_lidar_info = dict()
            sweep_lidar_info['sample_token'] = info['sample_token']
            sweep_lidar_info['ego_pose'] ={'token': info['sample_token'],
                                          'timestamp': data_frames['name'][0:8],
                                          'rotation': ego_rotation,
                                          'translation': ego_translation}
            sweep_lidar_info['timestamp'] = info['timestamp']
            sweep_lidar_info['filename'] = 'lidar/' + data_frames['videoName'] + '/' + data_frames['name'][0:8] + '_lidar_center'
            sweep_lidar_info['calibrated_sensor'] = {'token': info['sample_token'] + '_' + cam_name,
                                                     'sensor_token': info['sample_token'] + '_' + 'LIDAR_TOP',
                                                     'translation': ego_translation,
                                                     'rotation': ego_rotation,
                                                     'camera_intrinsic': []}
            lidar_infos[lidar_name] = sweep_lidar_info

        lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
        cam_sweeps = [dict() for _ in range(max_cam_sweeps)]

        # Remove empty sweeps.
        for i, sweep in enumerate(cam_sweeps):
            if len(sweep.keys()) == 0:
                cam_sweeps = cam_sweeps[:i]
                break
        for i, sweep in enumerate(lidar_sweeps):
            if len(sweep.keys()) == 0:
                lidar_sweeps = lidar_sweeps[:i]
                break

        info['cam_infos'] = cam_infos
        info['cams'] = cam_infos
        info['lidar2ego_translation'] = []
        info['lidar2ego_rotation'] = []
        info['ego2global_translation'] = ego_translation
        info['ego2global_rotation'] = ego_rotation
        info['timestamp'] = data_frames['name'][0:8]
        info['lidar_infos'] = lidar_infos

        info['cam_sweeps'] = cam_sweeps
        info['lidar_sweeps'] = lidar_sweeps
        ann_infos = list()
        # bevdepth 版本
        # for cam_name in cam_names:
        #     data_frames = infos_all[cam_name]['frames'][index]
        #     for ann in data_frames['labels']:
        #         ann_info = dict()
        #         ann_info['token'] = info['sample_token']
        #         ann_info['sample_token'] = info['sample_token']
        #         ann_info['instance_token'] = ann['id']
        #         ann_info['visibility_token'] = '4'
        #         ann_info['attribute_tokens'] = ann['category']
        #         ann_info['translation'] = ann['box3d']['location']
        #         ann_info['size'] = ann['box3d']['dimension']
        #
        #         ann_info['rotation'] = ann['box3d']['orientation'] # xyz2Quaternion(ann['box3d']['orientation'], euler='xyz')
        #         ann_info['prev'] = ''
        #         ann_info['next'] = ''
        #         ann_info['num_lidar_pts'] = 1
        #         ann_info['num_radar_pts'] = 0
        #         ann_info['category_name'] = ann['attributes']['type']
        #         ann_info['category'] = ann['category']
        #         ann_info['velocity'] = np.array([0.0,  0.0, 0.0])
        #         ann_infos.append(ann_info)
        # bevdet v2
        gt_boxes = list()
        gt_labels = list()
        taken = list()
        for cam_name in cam_names:
            data_frames = infos_all[cam_name]['frames'][index]
            for ann in data_frames['labels']:
                if ann['id'] in taken:
                    continue
                ann_info = dict()
                ann_info['token'] = info['sample_token']
                ann_info['sample_token'] = info['sample_token']
                ann_info['instance_token'] = ann['id']
                ann_info['visibility_token'] = '4'
                ann_info['attribute_tokens'] = ann['category']
                ann_info['translation'] = ann['box3d']['location']
                ann_info['size'] = ann['box3d']['dimension']
                ann_info['orientation'] = ann['box3d']['orientation']
                tmp = data_frames['extrinsics']['location']
                ann_info['word2cam_translate'] = [tmp[1], tmp[2], tmp[0]]
                ann_info['word2cam_rotate'] = data_frames['extrinsics']['rotation'][2]
                ann_info['ego_rotation'] = ego_rotation
                ann_info['ego_translation'] = ego_translation
                height, width, length = ann_info['size']
                wlh = [height, length, width]
                # print('*********************************')
                # print('ann_info[translation]', ann_info['translation'])
                # print(np.array(ann_info['translation']))
                # print(np.array(ann_info['word2cam_translate']))
                # print(np.array(ann_info['ego_translation']))
                # print(np.array(ann_info['translation']) + np.array(ann_info['word2cam_translate'])-np.array(ann_info['ego_translation']))
                box = Box(
                    np.array(ann_info['translation']),
                    wlh,
                    Quaternion(ann_info['orientation'][1]),  # 无效
                    velocity=np.array([0.0, 0.0, 0.0]),
                )
                box.rotate(Quaternion(matrix=pitch2matrix_shift_box(math.radians(ann_info['word2cam_rotate']))))
                box.translate(np.array(ann_info['word2cam_translate']))
                box.translate(-np.array(ego_translation))
                box.rotate(Quaternion(
                    matrix=pitch2matrix_shift_box(math.radians(math.radians(ann_info['ego_rotation'])))).inverse)
                # print('111111111111111111111111', box.center)
                # 相机坐标系→世界坐标系
                # print(cam_name)
                # print('loca', data_frames['extrinsics']['location'])
                # print('rot', data_frames['extrinsics']['rotation'])
                # box.translate(-np.array(data_frames['extrinsics']['location']))
                # box.rotate(Quaternion(data_frames['extrinsics']['rotation'][2]+0.0000000001).inverse)
                # print('translate', -np.array(data_frames['extrinsics']['location']))
                # print('2222222222222222222222222222222', box.center)
                # 世界坐标系→自车坐标系
                # box.rotate(Quaternion(ego_rotation + 0.0000000001))
                # box.translate(np.array(ego_translation))
                box_xyz = np.array(box.center)
                box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
                box_yaw = ann_info['orientation'][1] + math.radians(ann_info['word2cam_rotate']) - math.radians(
                    ann_info['ego_rotation'])
                box_yaw = yaw_lyft2nus(box_yaw)
                box_velo = np.array(box.velocity[:2])
                # print('translate', ego_translation)
                # print('3333333333333333333333', box.center)
                # print('box_xyz', box_xyz)
                # print('box_dxdydz', box_dxdydz)
                # print('box_yaw', box_yaw)
                # print('box_velo', box_velo)
                if box_xyz[0] > 60 or box_xyz[0] < -60:
                    continue
                if box_xyz[2] > 60 or box_xyz[2] < -60:
                    continue
                gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
                gt_boxes.append(gt_box)
                gt_labels.append(
                    classes.index(
                        map_name_from_general_to_detection[ann['category']]))
                ann_info['rotation'] = ann['box3d'][
                    'orientation']  # xyz2Quaternion(ann['box3d']['orientation'], euler='xyz')
                ann_info['prev'] = ''
                ann_info['next'] = ''
                ann_info['num_lidar_pts'] = 1
                ann_info['num_radar_pts'] = 0
                ann_info['cam_name'] = cam_name
                ann_info['category_name'] = ann['attributes']['type']
                ann_info['category'] = ann['category']
                ann_info['velocity'] = np.array([0.0, 0.0, 0.0])
                ann_infos.append(ann_info)
                taken.append(ann['id'])
        info['ann_infos'] = gt_boxes, gt_labels
        info['ann_infos_v1'] = ann_infos
        infos_new.append(info)
        index = index + 1
    info_all = dict()
    info_all['infos'] = infos_new
    # info_all['infos'] = infos_old['metadata']

    return info_all


def main():
    det_3d_path_train = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/left_stereo/det_3d.json'
    infos_train = mmcv.load(det_3d_path_train)
    # infos_train['config']['categories']
    # [{'name': 'pedestrian'}, {'name': 'car'}, {'name': 'truck'}, {'name': 'bus'}, {'name': 'motorcycle'}, {'name': 'bicycle'}]
    infos_new = info_transfor(infos_train)
    mmcv.dump(infos_new, '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/PKL/bevdetv2_shift_infos_Train_v1.pkl')



if __name__ == '__main__':
    main()

