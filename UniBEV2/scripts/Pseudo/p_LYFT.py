# Copyright (c) Phigent Robotics. All rights reserved.
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion
from scipy.spatial.transform import Rotation


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def points2depthmap(points, height, width):
    height, width = height, width
    depth_map = torch.zeros((height, width), dtype=torch.float32)
    coor = torch.round(points[:, :2] )
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                    depth < 200) & (depth >= -200)
    coor, depth = coor[kept1], depth[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth
    return depth_map


def point2img(information, index, view, intrins, lidar2lidarego, save_path, cams_name):
    # cam2ego
    cam2camego = np.eye(4, dtype=np.float32)
    cam2camego[:3, :3] = Quaternion(information[index]['cams'][view]['sensor2ego_rotation']).rotation_matrix
    cam2camego[:3, 3] = information[index]['cams'][view]['sensor2ego_translation']
    cam2camego = cam2camego
    cam2camego = torch.from_numpy(cam2camego).float()
    # intric
    intrins = torch.from_numpy(intrins).float()
    cam2img = np.eye(4, dtype=np.float32)
    cam2img = torch.from_numpy(cam2img)
    cam2img[:3, :3] = intrins
    # lidar2lidarego
    lidar2lidarego = torch.from_numpy(np.array(lidar2lidarego)).float()
    # lidar2cam
    lidar2cam = torch.inverse(cam2camego).matmul(lidar2lidarego)
    # point
    lidar_points = np.fromfile(information[index]['lidar_path'], dtype=np.float32)
    lidar_points = lidar_points.reshape(-1, 5)[:, :3]
    points_lidar = torch.from_numpy(lidar_points).float()
    # lidar2img
    lidar2img = cam2img.matmul(lidar2cam)
    # trans
    points_img = points_lidar.matmul(
        lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    points_img = torch.cat(
        [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
        1)
    depth_map = points2depthmap(points_img, image_with, image_height)
    # save map
    depth_mapnumpy = depth_map.numpy()
    depth_mapnumpy = depth_mapnumpy + 0.001
    img = cv2.imread(information[index]['cams'][view]['data_path'])
    depth_mapnumpy = depth_mapnumpy / np.max(np.max(depth_mapnumpy))
    depth_mapped = (depth_mapnumpy * 255).astype(np.uint8)  # 将归一化深度值映射到 [0, 255] 范围，并转换为无符号8位整数类型
    color_image = cv2.applyColorMap(depth_mapped, cv2.COLORMAP_JET)
    img = img * 0.2 + color_image * 0.8
    # cv2.imwrite(os.path.join(save_path, information[index]['token']+'_Depth'+cams_name+'.jpg'), color_image)
    cv2.imwrite(os.path.join(save_path, 'LYFT'+information[index]['token'][1:20]+'_DepthFusion'+cams_name+'.jpg'), img)
    return depth_map



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


def ego2img(points_ego, camrera_info):
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
scale_factor = 2
canva_size = 1000
show_range = 50
vis_frames = 50
draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                               (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                               (2, 6), (3, 7)]
color_map = {0: (255, 255, 0), 1: (0, 255, 255)}

info_path = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/data/lyft/lyft_infos_ori_ann_val.pkl'
dataset = pickle.load(open(info_path, 'rb'))
save_path = '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/VIS_PTS'
print('saving visualized result to %s' % save_path)

list = [dataset['infos'][i*100] for i in range(10)]
Quaternion(list[0]['lidar2ego_rotation']).rotation_matrix
# array([[-9.98455841e-01, -5.07202299e-02, -2.26581327e-02],
#        [ 5.07071750e-02, -9.98712901e-01,  1.15070678e-03],
#        [-2.26873336e-02,  5.77763453e-16,  9.99742609e-01]])
list[0]['lidar2ego_translation']


from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
import torch
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)

draw_gaussian = draw_heatmap_gaussian
downsample=16
# 读取 gt_boxes = infos['ann_infos'][0]
infos = dataset['infos'][123]
gt_boxes = infos['ann_infos'][0]
gt_label = infos['ann_infos'][1]
gt_boxes = np.array(gt_boxes)
gt_boxes = gt_boxes[:, 0:7]
length, width = gt_boxes[:, 3], gt_boxes[:, 4]
num_boxes, _ = gt_boxes.shape
img = cv2.imread(infos['cams']['CAM_FRONT_LEFT']['data_path'])
for cam_name in views:
    # 投影到不同平面x，y
    gravity_center_lidar_gt = LB(gt_boxes, origin=(0.5, 0.5, 0.5)).gravity_center.numpy()
    gravity_center_img, valid = ego2img(gravity_center_lidar_gt, infos['cams'][cam_name])
    corners_lidar_gt = LB(gt_boxes, origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
    corners_lidar_img, valid1 = ego2img(corners_lidar_gt, infos['cams'][cam_name])
    corners_lidar_img = corners_lidar_img.reshape(num_boxes, 8, 2)
    box_width = corners_lidar_img[:, :, 0].max(1) - corners_lidar_img[:, :, 0].min(1)
    box_height = corners_lidar_img[:, :, 1].max(1) - corners_lidar_img[:, :, 1].min(1)
    # 判断有效性
    valid = np.logical_and(valid, check_point_in_img(gravity_center_img, img.shape[0], img.shape[1]))
    # 生成heatmap
    gravity_center_img = torch.tensor(gravity_center_img)
    heatmap = gravity_center_img.new_zeros(12, img.shape[0], img.shape[1])
    center_int = gravity_center_img.to(torch.int32)
    for center_index in range(len(center_int)):
        if valid[center_index]==True:
            radius = gaussian_radius((torch.tensor(box_width[center_index]), torch.tensor(box_height[center_index])), min_overlap=0.1)
            radius = max(2*downsample, int(radius))
            radius = min(radius, 200)
            draw_gaussian(heatmap[gt_label[center_index]], [center_int[center_index, 0], center_int[center_index, 1]], radius)
    # heatmap[:, 72, 549]
    img[:, :, 0] = img[:, :, 0]*0.5 + 50 * heatmap[0, :, :].numpy()
    img[:, :, 1] = img[:, :, 1]*0.5 #+ 50* heatmap[1, :, :].numpy()
    img[:, :, 2] = img[:, :, 2]*0.5 #+ 50* heatmap[2, :, :].numpy()
    cv2.imwrite('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/scripts/Pseudo/'+cam_name+'asd.png', img)



# for cnt, infos in enumerate(list):
#     if cnt % 10 == 0:
#         print('%d/%d' % (cnt, min(vis_frames, len(dataset['infos']))))
#     gt_boxes = infos['ann_infos'][0]
#     gt_boxes = np.array(gt_boxes)
#     gt_boxes = gt_boxes[:, 0:7]
#
#     corners_lidar_gt = \
#         LB(gt_boxes,
#            origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
#
#     imgs = []
#     for view in views:
#         img = cv2.imread(infos['cams'][view]['data_path'])
#         # draw instances
#         corners_img, valid = ego2img(corners_lidar_gt, infos['cams'][view])
#         # print(view)
#         # print(infos['cams'][view]['sensor2lidar_rotation'])
#         # print(Quaternion(infos['cams'][view]['sensor2ego_rotation']).rotation_matrix)
#         # print(infos['cams'][view]['sensor2lidar_translation'])
#         # print(infos['cams'][view]['sensor2ego_translation'])
#         valid = np.logical_and(
#             valid,
#             check_point_in_img(corners_img, img.shape[0], img.shape[1]))
#         valid = valid.reshape(-1, 8)
#         corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)
#         for aid in range(valid.shape[0]):
#             for index in draw_boxes_indexes_img_view:
#                 if valid[aid, index[0]] and valid[aid, index[1]]:
#                     cv2.line(
#                         img,
#                         corners_img[aid, index[0]],
#                         corners_img[aid, index[1]],
#                         color=color_map[1],
#                         thickness=scale_factor)
#         imgs.append(img)
#         image_height = img.shape[1]
#         image_with = img.shape[0]
#         print(img.shape)
#         print(infos['cams'][view]['data_path'])
#         print(infos['token'][1:20])
#         print(infos['cams'][view]['cam_intrinsic'])
#     # bird-eye-view
#     canvas = np.zeros((int(canva_size), int(canva_size), 3),
#                       dtype=np.uint8)
#     # draw lidar points
#     lidar_points = np.fromfile(infos['lidar_path'], dtype=np.float32)
#     lidar_points = lidar_points.reshape(-1, 5)[:, :3]
#     lidar_points = lidar_points[:, [1, 0, 2]]
#     # lidar_points[:, 1] = -lidar_points[:, 1]
#     lidar_points[:, :2] = \
#         (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
#     for p in lidar_points:
#         if check_point_in_img(
#                 p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
#             color = depth2color(p[2])
#             cv2.circle(
#                 canvas, (int(p[0]), int(p[1])),
#                 radius=0,
#                 color=color,
#                 thickness=1)
#     # draw instance
#     corners_lidar_gt = corners_lidar_gt.reshape(-1, 8, 3)
#     # print(corners_lidar_gt)
#     corners_lidar_gt = corners_lidar_gt[:, :, [1, 0, 2]]
#     corners_lidar_gt[:, :, 0] = -corners_lidar_gt[:, :, 0]
#     corners_lidar_gt[:, :, 1] = -corners_lidar_gt[:, :, 1]
#     bottom_corners_bev = corners_lidar_gt[:, [0, 3, 7, 4], :2]
#     bottom_corners_bev = \
#         (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
#     bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
#     center_bev = corners_lidar_gt[:, [0, 3, 7, 4], :2].mean(axis=1)
#     head_bev = corners_lidar_gt[:, [0, 4], :2].mean(axis=1)
#     canter_canvas = \
#         (center_bev + show_range) / show_range / 2.0 * canva_size
#     center_canvas = canter_canvas.astype(np.int32)
#     head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
#     head_canvas = head_canvas.astype(np.int32)
#     for rid in range(len(center_canvas)):
#         score = 0
#         score = min(1 * 2.0, 1.0)
#         color = color_map[0]
#         for index in draw_boxes_indexes_bev:
#             cv2.line(
#                 canvas,
#                 bottom_corners_bev[rid, index[0]],
#                 bottom_corners_bev[rid, index[1]],
#                 [color[0] * score, color[1] * score, color[2] * score],
#                 thickness=1)
#         cv2.line(
#             canvas,
#             center_canvas[rid],
#             head_canvas[rid],
#             [color[0] * score, color[1] * score, color[2] * score],
#             1,
#             lineType=8)
#     # fuse image-view and bev
#     img = np.zeros((image_with * 2 + canva_size * scale_factor, image_height * 3, 3),
#                    dtype=np.uint8)
#     img[:image_with, :, :] = np.concatenate(imgs[:3], axis=1)
#     img_back = np.concatenate(
#         [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
#         axis=1)
#     img[image_with + canva_size * scale_factor:, :, :] = img_back
#     img = cv2.resize(img, (int(image_height / scale_factor * 3),
#                            int(image_with / scale_factor * 2 + canva_size)))
#     w_begin = int((image_height * 3 / scale_factor - canva_size) // 2)
#     img[int(image_with / scale_factor):int(image_with / scale_factor) + canva_size,
#     w_begin:w_begin + canva_size, :] = canvas
#     cv2.imwrite(os.path.join(save_path, 'LYFT'+infos['token'][1:20]+'.jpg'), img)
#     cams_name = 'CAM_FRONT'
#     intrins = infos['cams'][cams_name]['cam_intrinsic']
#     lidar2lidarego = np.eye(4, dtype=np.float32)
#     lidar2lidarego[:3, :3] = Quaternion(list[cnt]['lidar2ego_rotation']).rotation_matrix
#     lidar2lidarego[:3, 3] = list[cnt]['lidar2ego_translation']
#     point2img(list, cnt, cams_name, intrins, lidar2lidarego, save_path, cams_name)
#     cams_name = 'CAM_BACK'
#     intrins = infos['cams'][cams_name]['cam_intrinsic']
#     lidar2lidarego = np.eye(4, dtype=np.float32)
#     lidar2lidarego[:3, :3] = Quaternion(list[cnt]['lidar2ego_rotation']).rotation_matrix
#     lidar2lidarego[:3, 3] = list[cnt]['lidar2ego_translation']
#     point2img(list, cnt, cams_name, intrins, lidar2lidarego, save_path, cams_name)
