# Copyright (c) Phigent Robotics. All rights reserved.
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(np.int)
    diff = (gray - rank / num_rank) * num_rank
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        infos['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego


views = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
    'CAM_BACK', 'CAM_BACK_RIGHT']
scale_factor = 4
canva_size = 1000
show_range = 50
vis_frames = 50
draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                               (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                               (2, 6), (3, 7)]
color_map = {0: (255, 255, 0), 1: (0, 255, 255)}

info_path = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl'
dataset = pickle.load(open(info_path, 'rb'))
save_path = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/VIS_PTS'
print('saving visualized result to %s' % save_path)

list = [dataset['infos'][i*100] for i in range(10)]


# sensor2ego_rotation
# sensor2ego_translation
# ego2global_rotation
# ego2global_translation

## 'dataset['infos'][1]'
# dict_keys(['lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation',
# 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts',
# 'num_radar_pts', 'valid_flag', 'ann_infos', 'scene_token'])
# dataset['infos'][1]['ego2global_translation']
# [249.87286140205782, 917.5586761485395, 0.0]
# dataset['infos'][1]['ego2global_rotation']
# [0.998352367217269, -0.008540212772537494, 0.0022935935593023957, -0.05669528257323947]
# dataset['infos'][1]['cams']['CAM_FRONT_RIGHT']['sensor2ego_rotation']
# [0.2060347966337182, -0.2026940577919598, 0.6824507824531167, -0.6713610884174485]
# dataset['infos'][1]['cams']['CAM_FRONT_RIGHT']['ego2global_translation']


# dataset['infos'][1]['gt_boxes'][0]
# [-7.29056168, -8.70706277, -1.09857236,  4.257     ,  1.726     ,
#    1.489     ,  0.34336675]
# dataset['infos'][1]['ann_infos'][0][0]
# [-7.80247049,  7.27846286,  0.99478929,  4.257     ,  1.726     ,
#         1.489     , -1.22539256,  0.1748109 , -0.50812598])


# for cnt, infos in enumerate(dataset['infos'][500:min(vis_frames, len(dataset['infos']))]):
for cnt, infos in enumerate(list):
    if cnt % 10 == 0:
        print('%d/%d' % (cnt, min(vis_frames, len(dataset['infos']))))
    ## collect instances
    # pred_res = res['results'][infos['token']]
    # pred_boxes = [
    #     pred_res[rid]['translation'] + pred_res[rid]['size'] + [
    #         Quaternion(pred_res[rid]['rotation']).yaw_pitch_roll[0] +
    #         np.pi / 2
    #     ] for rid in range(len(pred_res))
    # ]
    # if len(pred_boxes) == 0:
    #     corners_lidar = np.zeros((0, 3), dtype=np.float32)
    # else:
    #     pred_boxes = np.array(pred_boxes, dtype=np.float32)
    #     boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))
    #     corners_global = boxes.corners.numpy().reshape(-1, 3)
    #     corners_global = np.concatenate(
    #         [corners_global,
    #          np.ones([corners_global.shape[0], 1])],
    #         axis=1)
    #     l2g = get_lidar2global(infos)
    #     corners_lidar = corners_global @ np.linalg.inv(l2g).T
    #     corners_lidar = corners_lidar[:, :3]
    # pred_flag = np.ones((corners_lidar.shape[0] // 8,), dtype=np.bool)
    # scores = [
    #     pred_res[rid]['detection_score'] for rid in range(len(pred_res))
    # ]
    gt_boxes = infos['gt_boxes']
    gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
    width = gt_boxes[:, 4].copy()
    gt_boxes[:, 4] = gt_boxes[:, 3]
    gt_boxes[:, 3] = width

    print(infos['gt_boxes'])
    break
    corners_lidar_gt = \
        LB(infos['gt_boxes'],
           origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
    # corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt],
    #                                axis=0)
    # gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool)
    # pred_flag = np.concatenate(
    #     [pred_flag, np.logical_not(gt_flag)], axis=0)
    # scores = scores + [0 for _ in range(infos['gt_boxes'].shape[0])]
    # scores = np.array(scores, dtype=np.float32)
    # sort_ids = np.argsort(scores)
    # image view

    imgs = []
    for view in views:
        img = cv2.imread(infos['cams'][view]['data_path'])
        # draw instances
        corners_img, valid = lidar2img(corners_lidar_gt, infos['cams'][view])
        valid = np.logical_and(
            valid,
            check_point_in_img(corners_img, img.shape[0], img.shape[1]))
        valid = valid.reshape(-1, 8)
        corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)
        for aid in range(valid.shape[0]):
            for index in draw_boxes_indexes_img_view:
                if valid[aid, index[0]] and valid[aid, index[1]]:
                    cv2.line(
                        img,
                        corners_img[aid, index[0]],
                        corners_img[aid, index[1]],
                        color=color_map[1],
                        thickness=scale_factor)
        imgs.append(img)
    # bird-eye-view
    canvas = np.zeros((int(canva_size), int(canva_size), 3),
                      dtype=np.uint8)
    # draw lidar points
    lidar_points = np.fromfile(infos['lidar_path'], dtype=np.float32)
    lidar_points = lidar_points.reshape(-1, 5)[:, :3]
    lidar_points[:, 1] = -lidar_points[:, 1]
    lidar_points[:, :2] = \
        (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
    for p in lidar_points:
        if check_point_in_img(
                p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
            color = depth2color(p[2])
            cv2.circle(
                canvas, (int(p[0]), int(p[1])),
                radius=0,
                color=color,
                thickness=1)
    # draw instances
    corners_lidar_gt = corners_lidar_gt.reshape(-1, 8, 3)
    corners_lidar_gt[:, :, 1] = -corners_lidar_gt[:, :, 1]
    bottom_corners_bev = corners_lidar_gt[:, [0, 3, 7, 4], :2]
    bottom_corners_bev = \
        (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
    bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
    center_bev = corners_lidar_gt[:, [0, 3, 7, 4], :2].mean(axis=1)
    head_bev = corners_lidar_gt[:, [0, 4], :2].mean(axis=1)
    canter_canvas = \
        (center_bev + show_range) / show_range / 2.0 * canva_size
    center_canvas = canter_canvas.astype(np.int32)
    head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
    head_canvas = head_canvas.astype(np.int32)
    for rid in range(len(center_canvas)):
        score = 0
        score = min(1 * 2.0, 1.0)
        color = color_map[0]
        for index in draw_boxes_indexes_bev:
            cv2.line(
                canvas,
                bottom_corners_bev[rid, index[0]],
                bottom_corners_bev[rid, index[1]],
                [color[0] * score, color[1] * score, color[2] * score],
                thickness=1)
        cv2.line(
            canvas,
            center_canvas[rid],
            head_canvas[rid],
            [color[0] * score, color[1] * score, color[2] * score],
            1,
            lineType=8)
    # fuse image-view and bev
    img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3),
                   dtype=np.uint8)
    img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
    img_back = np.concatenate(
        [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
        axis=1)
    img[900 + canva_size * scale_factor:, :, :] = img_back
    img = cv2.resize(img, (int(1600 / scale_factor * 3),
                           int(900 / scale_factor * 2 + canva_size)))
    w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
    img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
    w_begin:w_begin + canva_size, :] = canvas
    cv2.imwrite(os.path.join(save_path, '%s.jpg' % infos['token']), img)
