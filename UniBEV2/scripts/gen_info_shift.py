import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import math
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

translation = {'front': [0.5, 0.0, 0.0],
               'left_stereo': [],
               'left_45': [],
               'left_90': [],
               'right_45': [],
               'right_90': []

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


# cam_names = [
#             'front', 'left_stereo', 'left_45', 'left_90',
#             'right_45', 'right_90'
#         ]

cam_names = [
            'left_45'
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
        sweep_cam_info = dict()
        cam_datas = list()
        lidar_datas = list()
        info['sample_token'] = data_frames['videoName'] + '_' + data_frames['name'][0:8]
        info['timestamp'] = data_frames['name'][0:8]
        info['scene_token'] = data_frames['attributes']

        lidar_names = ['LIDAR_TOP']
        cam_infos = dict()
        lidar_infos = dict()
        cam_datas = list()
        ego_translation = list(map(lambda x: (x[0] + x[1])/2,
                 zip(infos_all['left_90']['frames'][index]['extrinsics']['location'], infos_all['right_90']['frames'][index]['extrinsics']['location'])))


        ego_rotation = xyz2Quaternion(infos_all['front']['frames'][index]['extrinsics']['rotation'])
        for cam_name in cam_names:
            # cam_data = nusc.get('sample_data',
            #                     cur_sample['data'][cam_name])
            # cam_datas.append(cam_data)
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


            sweep_cam_info['filename'] = cam_name + '/' + data_frames['videoName'] + '/' + infos_all[cam_name]['frames'][index]['name']

            sweep_cam_info['calibrated_sensor'] = {'token': info['sample_token'] + '_' + cam_name,
                                                   'sensor_token': cam_name,
                                                   'translation': infos_all[cam_name]['frames'][index]['extrinsics']['location'],
                                                   'rotation': xyz2Quaternion(infos_all[cam_name]['frames'][index]['extrinsics']['rotation']),
                                                   'camera_intrinsic': [[640.0, 0.0, 640.0], [0.0, 640.0, 400.0],
                                                                        [0.0, 0.0, 1.0]]}

            cam_infos[cam_name] = sweep_cam_info

        for lidar_name in lidar_names: #
            # lidar_data = nusc.get('sample_data',
            #                       cur_sample['data'][lidar_name])
            # lidar_datas.append(lidar_data)
            sweep_lidar_info = dict()
            sweep_lidar_info['sample_token'] = info['sample_token']
            sweep_lidar_info['ego_pose'] ={'token': info['sample_token'],
                                          'timestamp': data_frames['name'][0:8],
                                          'rotation': ego_rotation,
                                          'translation': ego_translation}
            sweep_lidar_info['timestamp'] = info['timestamp']
            sweep_lidar_info['filename'] = 'lidar/' + data_frames['videoName'] + '/' + data_frames['name'][0:8] + '_lidar_center'
            sweep_lidar_info['calibrated_sensor'] = {'token': info['sample_token'] + '_' + cam_name,
                                                     'sensor_token': info['sample_token'] + '_' + 'LIDAR_TOP' ,
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
        info['lidar_infos'] = lidar_infos

        info['cam_sweeps'] = cam_sweeps
        info['lidar_sweeps'] = lidar_sweeps
        ann_infos = list()
        for cam_name in cam_names:
            data_frames = infos_all[cam_name]['frames'][index]
            for ann in data_frames['labels']:
                ann_info = dict()
                ann_info['token'] = info['sample_token']
                ann_info['sample_token'] = info['sample_token']
                ann_info['instance_token'] = ann['id']
                ann_info['visibility_token'] = '4'
                ann_info['attribute_tokens'] = ann['category']
                ann_info['translation'] = ann['box3d']['location']
                ann_info['size'] = ann['box3d']['dimension']

                ann_info['rotation'] = ann['box3d']['orientation'] # xyz2Quaternion(ann['box3d']['orientation'], euler='xyz')
                ann_info['prev'] = ''
                ann_info['next'] = ''
                ann_info['num_lidar_pts'] = 1
                ann_info['num_radar_pts'] = 0
                ann_info['category_name'] = ann['attributes']['type']
                ann_info['category'] = ann['category']
                ann_info['velocity'] = np.array([0.0,  0.0, 0.0])
                ann_infos.append(ann_info)
        info['ann_infos'] = ann_infos
        infos_new.append(info)
        index = index + 1

    return infos_new






def main():
    det_3d_path_train = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/left_stereo/det_3d.json'
    infos_train = mmcv.load(det_3d_path_train)
    infos_train = info_transfor(infos_train)
    mmcv.dump(infos_train, '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth/data/SHIFT/shift_infos_train_s.pkl')



if __name__ == '__main__':
    main()

