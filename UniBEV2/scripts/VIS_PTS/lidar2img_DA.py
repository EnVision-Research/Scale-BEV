# Copyright (c) Phigent Robotics. All rights reserved.
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion
from scipy.spatial.transform import Rotation
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
import torch


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


def lidar2img(points_lidar, camrera_info, cam_name):
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
    if cam_name == 'Camera_Back':
        camera2img = torch.Tensor([[560.16603057, 0.0, 800.0],
                               [0.0, 560.16603057, 450.0],
                               [0.0, 0.0, 1.0]])
    else:
        camera2img = torch.Tensor([[1142.5184053936916, 0.0, 800.0],
                               [0.0, 1142.5184053936916, 450.0],
                               [0.0, 0.0, 1.0]])
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def ego2img(points_ego, camrera_info, cam_name):
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
    if cam_name == 'Camera_Back':
        camera2img = np.array([[560.16603057, 0.0, 800.0],
                               [0.0, 560.16603057, 450.0],
                               [0.0, 0.0, 1.0]])
    else:
        camera2img = np.array([[1142.5184053936916, 0.0, 800.0],
                               [0.0, 1142.5184053936916, 450.0],
                               [0.0, 0.0, 1.0]])
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


def points2depthmap(points, height, width):
    height, width = height, width
    depth_map = torch.zeros((height, width), dtype=torch.float32)
    coor = torch.round(points[:, :2] )
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                    depth < 75) & (depth >= -75)
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

views = ['Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight', 'Camera_BackLeft', 'Camera_Back', 'Camera_BackRight']

scale_factor = 4
canva_size = 1000
show_range = 50
vis_frames = 50
draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                               (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                               (2, 6), (3, 7)]
color_map = {0: (255, 255, 0), 1: (0, 255, 255)}



info_path = './data/DeepAccident_data/bevdetv4-DeepAccident_infos_val.pkl'
dataset = pickle.load(open(info_path, 'rb'))
save_path = './scripts/VIS_PTS'
print('saving visualized result to %s' % save_path)

list = [dataset['infos'][i*97] for i in range(4)]

# list[0]['lidar_to_ego_matrix']
# [ 9.99999999e-01,  4.89380152e-09,  2.38924184e-09, -1.25964046e-05],
# [ 4.89380152e-09,  9.99999941e-01,  7.92610352e-11, 1.25666823e-05],
# [ 2.38924184e-09,  7.92610352e-11,  9.99999893e-01, 2.01074620e+00],
# [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]
#
# [ 9.99999968e-01, -1.60292198e-09,  6.04367129e-11, 2.28814221e-06],
# [-1.60292198e-09,  1.00000002e+00, -4.70699275e-13, 1.61999074e-05],
# [ 6.04367129e-11, -4.70699275e-13,  9.99999951e-01, 2.15065161e+00],
# [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]
#
# [ 9.99999956e-01,  1.88646225e-12, -1.19729409e-10, 5.77599576e-06],
# [ 1.88646225e-12,  1.00000000e+00, -1.58608362e-12, 1.35615651e-06],
# [-1.19729409e-10, -1.58608362e-12,  9.99999955e-01, 2.54514694e+00],
# [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]

# 参考nus构建了ego然后直接使用nus的lidar2ego_rotation，不需要调整点云0和1的顺序
# [-0.99998053,  0.00217566, -0.00584864],
# [ 0.00203327,  0.99970406,  0.02424172],
# [-0.00589965, -0.02422936,  0.99968902]]
# [0, 0, 0]
view = 'Camera_Front'
index = 4
lidar2lidarego = [[-0.99998053,  0.00217566, -0.00584864, 0],
                 [0.00203327,  0.99970406,  0.02424172, 0],
                 [-0.00589965, -0.02422936,  0.99968902, 0],
                 [0, 0, 0, 1]]


def point2img(information, index, view, intrins, lidar2lidarego, save_path):
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
    lidar_points = np.load(information[index]['lidar_path'])
    lidar_points = lidar_points['data']
    lidar_points = lidar_points.reshape(-1, 4)[:, :3]
    points_lidar = torch.from_numpy(lidar_points).float()
    # lidar2img
    lidar2img = cam2img.matmul(lidar2cam)
    # trans
    points_img = points_lidar.matmul(
        lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    points_img = torch.cat(
        [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
        1)
    depth_map = points2depthmap(points_img, 900, 1600)
    # save map
    depth_mapnumpy = depth_map.numpy()
    depth_mapnumpy = np.clip(depth_mapnumpy, 0.2, 50)
    depth_mapnumpy = ((depth_mapnumpy) / 75) * 255
    depth_mapnumpy = depth_mapnumpy.astype(np.uint8)
    color_image = cv2.applyColorMap(depth_mapnumpy, cv2.COLORMAP_JET)
    depth_map_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    depth_map_edges = cv2.Canny(depth_map_gray, 2, 255)
    depth_map_edges_color = cv2.cvtColor(depth_map_edges, cv2.COLOR_GRAY2BGR)
    depth_map_combined = cv2.addWeighted(color_image, 0.5, depth_map_edges_color, 0.5, 0)
    img = cv2.imread(information[index]['cams'][view]['image_path'])
    #depth_mapnumpy = depth_mapnumpy / np.max(np.max(depth_mapnumpy))
    #depth_mapped = (depth_mapnumpy * 255).astype(np.uint8)  # 将归一化深度值映射到 [0, 255] 范围，并转换为无符号8位整数类型

    img_Depth = img * 0.2 + color_image * 0.8
    img_Ege = img * 0.2 + depth_map_edges_color * 0.8
    # cv2.imwrite(os.path.join(save_path, 'DA'+information[index]['lidar_path'][-20:-4]+'_Depth'+view+'.jpg'), color_image)
    cv2.imwrite(os.path.join(save_path, 'DA'+information[index]['lidar_path'][-20:-4]+'_DepthFusion'+view+'.jpg'), img_Depth)
    cv2.imwrite(
        os.path.join(save_path, 'DA' + information[index]['lidar_path'][-20:-4] + '_EgeFusion' + view + '.jpg'), img_Ege)
    return depth_map






# for cnt, infos in enumerate(dataset['infos'][500:min(vis_frames, len(dataset['infos']))]):
for cnt, infos in enumerate(list):
    if cnt % 10 == 0:
        print('%d/%d' % (cnt, min(vis_frames, len(dataset['infos']))))
    gt_boxes = infos['ann_infos'][0]
    gt_boxes = np.array(gt_boxes)
    gt_boxes = gt_boxes[:, 0:7]
    # gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
    # gt_boxes = gt_boxes[:, 0:6]
    # width = gt_boxes[:, 4].copy()
    # gt_boxes[:, 4] = gt_boxes[:, 3]
    # gt_boxes[:, 3] = width
    # print(gt_boxes)
    corners_lidar_gt = \
        LB(gt_boxes,
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
    # print('lidar2ego_rotation', infos['lidar2ego_rotation'])

    imgs = []
    for view in views:
        img = cv2.imread(infos['cams'][view]['image_path'])
        # draw instances
        corners_img, valid = ego2img(corners_lidar_gt, infos['cams'][view], view)
        # print(view)
        # print(infos['cams'][view]['sensor2lidar_rotation'])
        # print(Quaternion(infos['cams'][view]['sensor2ego_rotation']).rotation_matrix)
        # print(infos['cams'][view]['sensor2lidar_translation'])
        # print(infos['cams'][view]['sensor2ego_translation'])
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
    breakpoint()
    # draw lidar points
    print(infos['lidar_path'])
    # lidar_points = _load_points(infos['lidar_path'])
    lidar_points = np.load(infos['lidar_path'])
    lidar_points = lidar_points['data']
    # # 查看文件中的数组名称
    # array_names = lidar_points.files
    # print("Arrays in .npz file:", array_names)
    # # 遍历每个数组并打印其内容
    # for array_name in array_names:
    #     array_data = lidar_points[array_name]
    #     print(f"Array '{array_name}':")
    #     print(array_data)
    print(lidar_points.shape)
    lidar_points = lidar_points.reshape(-1, 4)[:, :3]
    lidar_points = lidar_points[:, [1, 0, 2]]
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
    # draw instance
    corners_lidar_gt = corners_lidar_gt.reshape(-1, 8, 3)
    # print(corners_lidar_gt)
    corners_lidar_gt = corners_lidar_gt[:, :, [1, 0, 2]]
    corners_lidar_gt[:, :, 0] = -corners_lidar_gt[:, :, 0]
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
    cv2.imwrite(os.path.join(save_path, 'DA%s.jpg' % infos['lidar_path'][-20:-4]), img)
    # point2img
    lidar2lidarego = [[0.99998053, -0.00217566, 0.00584864, 0],
                      [-0.00203327, -0.99970406, -0.02424172, 0],
                      [-0.00589965, -0.02422936, 0.99968902, 0],
                      [0, 0, 0, 1]]
    lidar2lidarego = list[0]['lidar_to_ego_matrix']
    lidar2lidarego[1,:] = -lidar2lidarego[1,:]
    # lidar2lidarego[2, :] = -lidar2lidarego[2, :]
    # lidar2lidarego[0,:] = -1*lidar2lidarego[0,:]
    intrins = np.array([[1142.5184053936916, 0.0, 800.0],
                        [0.0, 1142.5184053936916, 450.0],
                        [0.0, 0.0, 1.0]])
    point2img(list, cnt, 'Camera_FrontLeft', intrins, lidar2lidarego, save_path)
    point2img(list, cnt, 'Camera_Front', intrins, lidar2lidarego, save_path)
    point2img(list, cnt, 'Camera_FrontRight', intrins, lidar2lidarego, save_path)
    point2img(list, cnt, 'Camera_BackLeft', intrins, lidar2lidarego, save_path)
    point2img(list, cnt, 'Camera_BackRight', intrins, lidar2lidarego, save_path)
    point2img(list, cnt, 'Camera_Back', intrins, lidar2lidarego, save_path)
    intrins = np.array([[560.16603057, 0.0, 800.0],
                           [0.0, 560.16603057, 450.0],
                           [0.0, 0.0, 1.0]])
    point2img(list, cnt, 'Camera_Back', intrins, lidar2lidarego, save_path)

