 # Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os
import torch
from PIL import Image
import copy
from pyquaternion import Quaternion
import math
import cv2
import numpy as np
from PIL import Image
from math import atan2, asin, cos, sin
from scipy.optimize import minimize
from scipy import io
from PIL import Image
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES
from .loading import PrepareImageInputs, PrepareImageInputs_DeepAccident
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from torchvision.transforms import ToTensor

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img






@PIPELINES.register_module()
class Load3DBoxesHeatmap_Uni(object):
    def __init__(self, classes, downsample_feature=8, aug_flag=True):
        self.classes = classes
        self.draw_gaussian = draw_heatmap_gaussian
        self.downsample = 16
        self.aug_flag = aug_flag
        self.classes_num = len(classes)
        self.downsample_feature = downsample_feature
        self.box_ann_num = 3+3+1
        self.c = np.sqrt(100.0 / np.square(450) + 100.0 / np.square(450))

        self.lyft_geo = []
        self.nus_geo = []
        self.waymo_geo = []
        self.num_count = 0
        self.save_flag = 1


    def img_transform_core(self, img, resize_dims, crop, flip, rotate, down):
        # adjust image
        img = img.resize(resize_dims, resample=Image.BICUBIC)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = img.resize((int((crop[2] - crop[0])/down), int((crop[3] - crop[1])/down)))
        return img

    def ego2img(self, points_ego, camrera_info, camera2img, geo_flag=False,resize=None):
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
            GeoMetric= self.get_GeoMetric(points_img)
            GeoMetric = GeoMetric * resize
        else:
            GeoMetric= None
        points_img = points_img / points_img[:, 2:3]
        points_img = points_img[:, :2]
        return points_img, valid, depth, GeoMetric

    def get_GeoMetric(self, points_img):
        points_img = points_img[::8,:]
        N = points_img.shape[0]
        if N >3:
            # random_index = np.random.choice(np.array(list(range(N))), 3, replace=False)
            # points_img = points_img[random_index, :]
            # A, B, C, D = self.get_3Dplane(points_img)
            A, B, C, D = self.fit_plane(points_img)
            GeoMetric=self.get_V_Metric(A, B, C, D)
        else:
            GeoMetric=100

        return GeoMetric

    def fit_plane(self, points):

        # 将 points 转换为 numpy 数组
        points = np.array(points)

        # 将坐标平移到原点
        centroid = np.mean(points, axis=0)
        points -= centroid

        # 计算协方差矩阵
        cov = np.cov(points.T)

        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # 找到特征值最小的特征向量，即为平面的法向量
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        # 将法向量单位化
        normal /= np.linalg.norm(normal)

        # 计算平面的偏移量
        offset = -np.dot(normal, centroid)

        # 将法向量和偏移量组合成一个四元组
        plane = tuple(np.append(normal, offset))

        # 将四元组中的元素重命名为 A, B, C, D
        A, B, C, D = plane

        return A, B, C, D

    def get_3Dplane(self, points_img):
        V1 = np.array(points_img[1,:]) - np.array(points_img[0,:])
        V2 = np.array(points_img[2,:]) - np.array(points_img[0,:])
        N = np.cross(V1, V2)
        A, B, C = N[0], N[1], N[2]
        D = -np.dot(N, points_img[0,:])
        return A, B, C, D

    def get_V_Metric(self, A, B, C, D, u0=352, d0=5, u1=352, d1=10):
        v0=(A*u0*d0+C*d0+D)/(B*d0)
        v1=(A*u1*d1+C*d1+D)/(B*d1)
        return abs(v1-v0)



    def check_point_in_img(self, points, height, width):
        valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
        valid = np.logical_and(
            valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
        return valid

    def get_2D_heatmap(self, results):
        gt_boxes, gt_labels = copy.deepcopy(results['ann_infos'])
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intrins_actuals = results['img_inputs']
        gt_boxes = np.array(gt_boxes)
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 9))
        # print('gt_boxes', gt_boxes.shape)
        gt_boxes = gt_boxes[:, 0:7]
        num_boxes, _ = gt_boxes.shape
        cam_names = results['cam_names']
        cam_data = results['curr']['cams'][cam_names[0]]

        if 'image_path' in cam_data.keys():
            filename = cam_data['image_path']
        else:
            filename = cam_data['data_path']
        img = Image.open(filename)
        width, height = img.size
        heatmaps = []
        ann_maps = []
        heatmap_masks = []
        gravity_center_lidar_gt = LB(gt_boxes, origin=(0.5, 0.5, 0.5)).gravity_center.numpy()
        corners_lidar_gt = LB(gt_boxes, origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)

        # print('??!!!!!!!!!????')
        label_num=0
        for cam_index in range(len(cam_names)):
            cam_name = cam_names[cam_index]
            # 3D box 对应增广： 修改center和长宽高
            [resize, resize_dims, crop, flip, rotate] = results['img_aug']['cam'+ str(cam_names.index(cam_name))][0]
            post_rot2 = results['img_aug']['cam'+ str(cam_names.index(cam_name))][1]
            post_tran2 = results['img_aug']['cam'+ str(cam_names.index(cam_name))][2]
            # print('post_rot2', resize)
            # print('post_rot2', post_rot2)
            # print('post_tran2', post_tran2)
            # 投影到不同平面x，y
            # 中心
            gravity_center_img, valid, depth, _ = self.ego2img(gravity_center_lidar_gt, results['curr']['cams'][cam_name],
                                                     np.array(intrins[cam_index]))
            # 边框
            corners_lidar_img, _, _, GeoMetric = self.ego2img(corners_lidar_gt, results['curr']['cams'][cam_name],
                                                np.array(intrins[cam_index]), geo_flag=True, resize=resize)
            if cam_index==0:
                FinalGeoMetric = GeoMetric
                # print('filename', filename)
                # print('GeoMetric', GeoMetric)
                if self.save_flag:
                    self.num_count = self.num_count + 1
                    if 'lyft' in filename:
                        self.lyft_geo.append(GeoMetric)
                    elif 'nuscenes' in filename:
                        self.nus_geo.append(GeoMetric)
                    else:
                        self.waymo_geo.append(GeoMetric)
                    if self.num_count%100==0:
                        io.savemat('/mnt/cfs/algorithm/hao.lu/Code/UniBEV/Stat/lyft_geo.mat', {'lyft_geo': self.lyft_geo})
                        io.savemat('/mnt/cfs/algorithm/hao.lu/Code/UniBEV/Stat/nus_geo.mat', {'nus_geo': self.nus_geo})
                        io.savemat('/mnt/cfs/algorithm/hao.lu/Code/UniBEV/Stat/waymo_geo.mat', {'waymo_geo': self.waymo_geo})


            corners_lidar_img = corners_lidar_img.reshape(num_boxes, 8, 2)
            box_width = corners_lidar_img[:, :, 0].max(1) - corners_lidar_img[:, :, 0].min(1)
            box_height = corners_lidar_img[:, :, 1].max(1) - corners_lidar_img[:, :, 1].min(1)
            # 判断有效性
            valid = np.logical_and(valid, self.check_point_in_img(gravity_center_img, height, width))
            gravity_center_img = (torch.tensor(gravity_center_img)).matmul(post_rot2[0:2, 0:2].T)
            act_intrins = np.array(intrins_actuals[cam_index, :, :])
            depth_real = torch.tensor(depth)
            depth = (depth * np.sqrt(
                100.0 / np.square(act_intrins[0][0]) + 100.0 / np.square(act_intrins[1][1]))) / self.c
            depth = torch.tensor(depth)
            gravity_center_img = gravity_center_img + post_tran2[0:2]
            gravity_center_img = gravity_center_img / self.downsample_feature
            box_width = resize * box_width / self.downsample_feature
            box_height = resize * box_height / self.downsample_feature
            height_act, width_act = crop[3] - crop[1], crop[2] - crop[0]
            # 生成heatmap
            heatmap = gravity_center_img.new_zeros(int(height_act / self.downsample_feature),
                                                   int(width_act / self.downsample_feature), self.classes_num)
            heatmap_mask = gravity_center_img.new_zeros(int(height_act / self.downsample_feature),
                                                        int(width_act / self.downsample_feature), self.classes_num)
            ann_map = gravity_center_img.new_zeros(int(height_act / self.downsample_feature),
                                                   int(width_act / self.downsample_feature), self.box_ann_num,
                                                   self.classes_num)
            center_int = gravity_center_img.to(torch.int32)
            # print(valid)
            for center_index in range(len(gt_labels)):
                label_num=label_num+1
                if (valid[center_index] == True) and (gt_labels[center_index] < self.classes_num):
                    radius = gaussian_radius(
                        (torch.tensor(box_width[center_index]), torch.tensor(box_height[center_index])),
                        min_overlap=0.1)
                    radius = max(1, int(radius))
                    radius = min(radius, int(40 / self.downsample_feature))
                    x_int, y_int = center_int[center_index, 0], center_int[center_index, 1]
                    self.draw_gaussian(heatmap[:, :, gt_labels[center_index]],
                                       [center_int[center_index, 0], center_int[center_index, 1]], radius)
                    heatmap_mask[y_int - 1:y_int + 2, x_int - 1:x_int + 2, gt_labels[center_index]] = 1
                    # heatmap_mask[y_int, x_int, gt_labels[center_index]] = 1
                    ann_map[y_int - 1:y_int + 2, x_int - 1:x_int + 2, 0, gt_labels[center_index]] = gravity_center_img[
                                                                                                        center_index, 0] - \
                                                                                                    center_int[
                                                                                                        center_index, 0]
                    ann_map[y_int - 1:y_int + 2, x_int - 1:x_int + 2, 1, gt_labels[center_index]] = gravity_center_img[
                                                                                                        center_index, 1] - \
                                                                                                    center_int[
                                                                                                        center_index, 1]
                    ann_map[y_int - 1:y_int + 2, x_int - 1:x_int + 2, 2, gt_labels[center_index]] = depth[center_index, 0]
                    ann_map[y_int - 1:y_int + 2, x_int - 1:x_int + 2, 3:6, gt_labels[center_index]] = torch.tensor(
                        gt_boxes[center_index, 3:6])
                    ann_map[y_int - 1:y_int + 2, x_int - 1:x_int + 2, 6, gt_labels[center_index]] = torch.tensor(
                        depth_real[center_index, 0])
                    # ann_map[y_int - 1:y_int + 1, x_int - 1:x_int + 2, 5, gt_labels[center_index]] = torch.sin(
                    #     torch.tensor(gt_boxes[center_index, 6])).unsqueeze(0)
                    # ann_map[y_int - 1:y_int + 1, x_int - 1:x_int + 2, 6, gt_labels[center_index]] = torch.cos(
                    #     torch.tensor(gt_boxes[center_index, 6])).unsqueeze(0)            heatmaps.append(heatmap)
            # print('label_num', label_num)
            # print('len(gt_labels)', len(gt_labels))
            heatmaps.append(heatmap)
            ann_maps.append(ann_map)
            heatmap_masks.append(heatmap_mask)

            #
            # # 对应增广
            # [resize, resize_dims, crop, flip, rotate] = results['img_aug']['cam'+ str(cam_names.index(cam_name))][0]
            # cam_data = results['curr']['cams'][cam_name]
            # if 'image_path' in cam_data.keys():
            #     filename = cam_data['image_path']
            # else:
            #     filename = cam_data['data_path']
            # img = Image.open(filename)
            # img.save('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/scripts/Pseudo/' + filename[-10:-5] + cam_name + 'ori.png')
            # img = self.img_transform_core(img, resize_dims, crop, flip, rotate, self.downsample_feature)
            # img = np.array(img)
            # print(img.shape)
            # heatmap_1 = np.array(heatmap)
            # print(heatmap_1.shape)
            # img[:, :, 0] = img[:, :, 0] * 0.5 + 100 * heatmap_1[:, :, 0]
            # img[:, :, 1] = img[:, :, 1] * 0.5 + 100 * heatmap_1[:, :, 5]
            # img[:, :, 2] = img[:, :, 2] * 0.5  # + 50* heatmap[2, :, :].numpy()
            # print(
            #     '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/scripts/Pseudo/'  + filename[-10:-5] + cam_name + '.png')
            # cv2.imwrite(
            #     '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/scripts/Pseudo/' + filename[-10:-5] + cam_name + '.png',
            #     img)
        heatmaps = torch.stack(heatmaps)
        ann_maps = torch.stack(ann_maps)
        heatmap_masks = torch.stack(heatmap_masks)
        results.update({"heatmaps_2d": heatmaps})
        results.update({"ann_maps_2d": ann_maps})
        results.update({"GeoMetric": FinalGeoMetric})
        results.update({"heatmap_masks_2d": heatmap_masks})
        return results




    def get_2D_aug_heatmap(self, results):
        gt_boxes, gt_labels = copy.deepcopy(results['ann_infos'])
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intrins_actuals = results['img_inputs']
        gt_boxes = np.array(gt_boxes)
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 9))
        # print('gt_boxes', gt_boxes.shape)
        gt_boxes = gt_boxes[:, 0:7]
        num_boxes, _ = gt_boxes.shape
        cam_names = results['cam_names']
        cam_data = results['curr']['cams'][cam_names[0]]

        if 'image_path' in cam_data.keys():
            filename = cam_data['image_path']
        else:
            filename = cam_data['data_path']
        img = Image.open(filename)
        width, height = img.size
        heatmaps = []
        ann_maps = []
        heatmap_masks = []
        gravity_center_lidar_gt = LB(gt_boxes, origin=(0.5, 0.5, 0.5)).gravity_center.numpy()
        corners_lidar_gt = LB(gt_boxes, origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
        for cam_index in range(len(cam_names)):
            cam_name = cam_names[cam_index]
            # 投影到不同平面x，y
            # 中心
            gravity_center_img, valid, depth, _ = self.ego2img(gravity_center_lidar_gt, results['bev_aug']['cam'+ str(cam_names.index(cam_name))],
                                                     np.array(intrins[cam_index]))
            # 边框
            corners_lidar_img, _, _, _ = self.ego2img(corners_lidar_gt, results['bev_aug']['cam'+ str(cam_names.index(cam_name))],
                                                np.array(intrins[cam_index]))
            corners_lidar_img = corners_lidar_img.reshape(num_boxes, 8, 2)
            box_width = corners_lidar_img[:, :, 0].max(1) - corners_lidar_img[:, :, 0].min(1)
            box_height = corners_lidar_img[:, :, 1].max(1) - corners_lidar_img[:, :, 1].min(1)
            # 判断有效性
            valid = np.logical_and(valid, self.check_point_in_img(gravity_center_img, height, width))
            # 3D box 对应增广： 修改center和长宽高
            [resize, resize_dims, crop, flip, rotate] = results['img_aug']['cam'+str(cam_names.index(cam_name))][0]
            post_rot2 = results['img_aug']['cam'+ str(cam_names.index(cam_name))][1]
            post_tran2 = results['img_aug']['cam'+ str(cam_names.index(cam_name))][2]
            # print('post_rot2', resize)
            # print('post_rot2', post_rot2)
            # print('post_tran2', post_tran2)
            gravity_center_img = (torch.tensor(gravity_center_img)).matmul(post_rot2[0:2, 0:2].T)
            act_intrins = np.array(intrins_actuals[cam_index, :, :])
            depth_real = torch.tensor(depth)
            depth = (depth * np.sqrt(
                100.0 / np.square(act_intrins[0][0]) + 100.0 / np.square(act_intrins[1][1]))) / self.c
            depth = torch.tensor(depth)
            gravity_center_img = gravity_center_img + post_tran2[0:2]
            gravity_center_img = gravity_center_img / self.downsample_feature
            box_width = resize * box_width / self.downsample_feature
            box_height = resize * box_height / self.downsample_feature
            height_act, width_act = crop[3] - crop[1], crop[2] - crop[0]

            # 生成heatmap
            heatmap = gravity_center_img.new_zeros(int(height_act / self.downsample_feature),
                                                   int(width_act / self.downsample_feature), self.classes_num)
            heatmap_mask = gravity_center_img.new_zeros(int(height_act / self.downsample_feature),
                                                        int(width_act / self.downsample_feature), self.classes_num)
            ann_map = gravity_center_img.new_zeros(int(height_act / self.downsample_feature),
                                                   int(width_act / self.downsample_feature), self.box_ann_num,
                                                   self.classes_num)
            center_int = gravity_center_img.to(torch.int32)
            for center_index in range(len(gt_labels)):
                if (valid[center_index] == True) and (gt_labels[center_index] < self.classes_num):
                    radius = gaussian_radius(
                        (torch.tensor(box_width[center_index]), torch.tensor(box_height[center_index])),
                        min_overlap=0.1)
                    radius = max(4, int(radius))
                    radius = min(radius, int(200 / self.downsample_feature))
                    x_int, y_int = center_int[center_index, 0], center_int[center_index, 1]
                    self.draw_gaussian(heatmap[:, :, gt_labels[center_index]],
                                       [center_int[center_index, 0], center_int[center_index, 1]], radius)
                    heatmap_mask[y_int - 1:y_int + 1, x_int - 1:x_int + 2, gt_labels[center_index]] = 1
                    ann_map[y_int - 1:y_int + 1, x_int - 1:x_int + 2, 0, gt_labels[center_index]] = gravity_center_img[
                                                                                                        center_index, 0] - \
                                                                                                    center_int[
                                                                                                        center_index, 0]
                    ann_map[y_int - 1:y_int + 1, x_int - 1:x_int + 2, 1, gt_labels[center_index]] = gravity_center_img[
                                                                                                        center_index, 1] - \
                                                                                                    center_int[
                                                                                                        center_index, 1]
                    ann_map[y_int - 1:y_int + 1, x_int - 1:x_int + 2, 2, gt_labels[center_index]] = depth[
                        center_index, 0]
                    ann_map[y_int - 1:y_int + 1, x_int - 1:x_int + 2, 3:6, gt_labels[center_index]] = torch.tensor(
                        gt_boxes[center_index, 3:6])
                    ann_map[y_int - 1:y_int + 1, x_int - 1:x_int + 2, 6, gt_labels[center_index]] = torch.tensor(depth_real[
                        center_index, 0])
                    # ann_map[y_int - 1:y_int + 1, x_int - 1:x_int + 2, 5, gt_labels[center_index]] = torch.sin(
                    #     torch.tensor(gt_boxes[center_index, 6])).unsqueeze(0)
                    # ann_map[y_int - 1:y_int + 1, x_int - 1:x_int + 2, 6, gt_labels[center_index]] = torch.cos(
                    #     torch.tensor(gt_boxes[center_index, 6])).unsqueeze(0)
            heatmaps.append(heatmap)
            ann_maps.append(ann_map)
            heatmap_masks.append(heatmap_mask)
            # #
            # # 对应增广
            # [resize, resize_dims, crop, flip, rotate] = results['img_aug'][cam_name][0]
            # cam_data = results['curr']['cams'][cam_name]
            # if 'image_path' in cam_data.keys():
            #     filename = cam_data['image_path']
            # else:
            #     filename = cam_data['data_path']
            # img = Image.open(filename)
            # img.save('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/scripts/Pseudo/' + cam_name + filename[-10:-4] + 'ori.png')
            # img = self.img_transform_core(img, resize_dims, crop, flip, rotate, self.downsample_feature)
            # img = np.array(img)
            # print(img.shape)
            # heatmap_1 = np.array(heatmap)
            # print(heatmap_1.shape)
            # img[:, :, 0] = img[:, :, 0] * 0.5 + 100 * heatmap_1[:, :, 0]
            # img[:, :, 1] = img[:, :, 1] * 0.5 + 100 * heatmap_1[:, :, 5]
            # img[:, :, 2] = img[:, :, 2] * 0.5  # + 50* heatmap[2, :, :].numpy()
            # print(
            #     '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/scripts/Pseudo/' + cam_name + filename[-10:-4] + '.png')
            # cv2.imwrite(
            #     '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/scripts/Pseudo/' + cam_name + filename[-10:-4] + '.png',
            #     img)
        heatmaps = torch.stack(heatmaps)
        ann_maps = torch.stack(ann_maps)
        heatmap_masks = torch.stack(heatmap_masks)
        results.update({"heatmaps_2d_aug": heatmaps})
        results.update({"ann_maps_2d_aug": ann_maps})
        results.update({"heatmap_masks_2d_aug": heatmap_masks})
        return results


    def __call__(self, results):
        results = self.get_2D_heatmap(results)
        if self.aug_flag:
            results = self.get_2D_aug_heatmap(results)
        return results




@PIPELINES.register_module()
class PrepareImageInputs_Uni(object):

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran


    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img


    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            # crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * (newH - fH) / 2)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            # crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * (newH - fH) / 2)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran

        # sweep ego to global
        if 'ego2global_rotation' in cam_info['cams'][cam_name].keys():
            w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
            sweepego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            sweepego2global_tran = torch.Tensor(
                cam_info['cams'][cam_name]['ego2global_translation'])
            sweepego2global = sweepego2global_rot.new_zeros((4, 4))
            sweepego2global[3, 3] = 1
            sweepego2global[:3, :3] = sweepego2global_rot
            sweepego2global[:3, -1] = sweepego2global_tran
        else:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
            sweepego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            sweepego2global_tran = torch.Tensor(
                np.array([0, 0, 0]))
            sweepego2global = sweepego2global_rot.new_zeros((4, 4))
            sweepego2global[3, 3] = 1
            sweepego2global[:3, :3] = sweepego2global_rot
            sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        if 'ego2global_rotation' in key_info['cams'][ego_cam].keys():
            w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
            keyego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            keyego2global_tran = torch.Tensor(
                key_info['cams'][ego_cam]['ego2global_translation'])
            keyego2global = keyego2global_rot.new_zeros((4, 4))
            keyego2global[3, 3] = 1
            keyego2global[:3, :3] = keyego2global_rot
            keyego2global[:3, -1] = keyego2global_tran
            global2keyego = keyego2global.inverse()
        else:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
            keyego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            keyego2global_tran = torch.Tensor(
                np.array([0, 0, 0]))
            keyego2global = keyego2global_rot.new_zeros((4, 4))
            keyego2global[3, 3] = 1
            keyego2global[:3, :3] = keyego2global_rot
            keyego2global[:3, -1] = keyego2global_tran
            global2keyego = keyego2global.inverse()

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego


        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info['cams'][cam_name]['sensor2ego_translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
                keyego2keysensor @ global2keyego @ sweepego2global
                @ sweepsensor2sweepego).inverse()
        return sweepsensor2keyego, keysensor2sweepsensor

    def quaternion_to_euler(self, q):
        # 将四元数转化为欧拉角
        # q: 四元数，[w, x, y, z]
        # 返回欧拉角，[pitch, yaw, roll]
        w, x, y, z = q
        pitch = asin(2 * (w * y - x * z))
        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        roll = atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        return np.array([pitch, yaw, roll])

    def euler_to_quaternion(self, e, aug_flag=True):
        # 将欧拉角转化为四元数
        # e: 欧拉角，[pitch, yaw, roll]
        # 返回四元数，[w, x, y, z]
        pitch, yaw, roll = e
        if aug_flag:
            pitch += np.random.uniform(*self.data_config['pitch_aug'])
            yaw += np.random.uniform(*self.data_config['yaw_aug']) # 这个幅度要大一点
            roll += np.random.uniform(*self.data_config['roll_aug'])
        # print(pitch, yaw, roll)
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        w = cy * cr * cp + sy * sr * sp
        x = cy * sr * cp - sy * cr * sp
        y = cy * cr * sp + sy * sr * cp
        z = sy * cr * cp - cy * sr * sp
        return np.array([w, x, y, z])

    def Euler_aug(self, q):
        e = self.quaternion_to_euler(q)
        q = self.euler_to_quaternion(e)
        return q


    def get_sensor2ego_transformation_aug(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name

        # sweep sensor to sweep ego
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        q = self.Euler_aug(np.array([w, x, y, z]))
        w, x, y, z = q
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        #print('ori sweepsensor2sweepego_tran', sweepsensor2sweepego_tran)
        sweepsensor2sweepego_tran = sweepsensor2sweepego_tran + torch.tensor([np.random.uniform(*self.data_config['extri_x_aug']), np.random.uniform(*self.data_config['extri_y_aug']), np.random.uniform(*self.data_config['extri_z_aug'])])
        #print('aft sweepsensor2sweepego_tran', sweepsensor2sweepego_tran)
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # print('w, x, y, z', w, x, y, z)
        # print('sweepsensor2sweepego_tran', sweepsensor2sweepego_tran)
        # print('sweepsensor2sweepego', sweepsensor2sweepego)
        # sweep ego to global
        if 'ego2global_rotation' in cam_info['cams'][cam_name].keys():
            w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
            sweepego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            sweepego2global_tran = torch.Tensor(
                cam_info['cams'][cam_name]['ego2global_translation'])
            sweepego2global = sweepego2global_rot.new_zeros((4, 4))
            sweepego2global[3, 3] = 1
            sweepego2global[:3, :3] = sweepego2global_rot
            sweepego2global[:3, -1] = sweepego2global_tran
        else:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
            sweepego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            sweepego2global_tran = torch.Tensor(
                np.array([0, 0, 0]))
            sweepego2global = sweepego2global_rot.new_zeros((4, 4))
            sweepego2global[3, 3] = 1
            sweepego2global[:3, :3] = sweepego2global_rot
            sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        if 'ego2global_rotation' in key_info['cams'][ego_cam].keys():
            w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
            keyego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            keyego2global_tran = torch.Tensor(
                key_info['cams'][ego_cam]['ego2global_translation'])
            keyego2global = keyego2global_rot.new_zeros((4, 4))
            keyego2global[3, 3] = 1
            keyego2global[:3, :3] = keyego2global_rot
            keyego2global[:3, -1] = keyego2global_tran
            global2keyego = keyego2global.inverse()
        else:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
            keyego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            keyego2global_tran = torch.Tensor(
                np.array([0, 0, 0]))
            keyego2global = keyego2global_rot.new_zeros((4, 4))
            keyego2global[3, 3] = 1
            keyego2global[:3, :3] = keyego2global_rot
            keyego2global[:3, -1] = keyego2global_tran
            global2keyego = keyego2global.inverse()

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego


        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info['cams'][cam_name]['sensor2ego_translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
                keyego2keysensor @ global2keyego @ sweepego2global
                @ sweepsensor2sweepego).inverse()
        return sweepsensor2keyego, keysensor2sweepsensor, q, sweepsensor2sweepego_tran


    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        rot_augs = []
        tran_augs = []
        intrins = []
        post_rots = []
        post_trans = []
        intrins_actuals = []
        if 'Camera_Front' in results['curr']['cams'].keys():
            self.ego_cam = 'Camera_Front'
            cam_names = ['Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight',
                         'Camera_BackLeft', 'Camera_Back', 'Camera_BackRight']
            results['cam_names'] = cam_names
        elif 'CAM_FRONT' in results['curr']['cams'].keys():
            self.ego_cam = 'CAM_FRONT'
            if 'CAM_BACK' in results['curr']['cams'].keys():
                cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT',  'CAM_BACK', 'CAM_BACK_RIGHT']
            else:
                cam_names = [
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
                    'CAM_BACK_RIGHT']

            results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        results.update(
            dict(img_aug=dict()))
        results.update(
            dict(bev_aug=dict()))
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            if 'image_path' in cam_data.keys():
                filename = cam_data['image_path']
            else:
                filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            if cam_name == 'Camera_Back':
                intrin = torch.Tensor([[560.16603057, 0.0, 800.0],
                                       [0.0, 560.16603057, 450.0],
                                       [0.0, 0.0, 1.0]])
            else:
                intrin = torch.Tensor([[1142.5184053936916, 0.0, 800.0],
                                       [0.0, 1142.5184053936916, 450.0],
                                       [0.0, 0.0, 1.0]])
            if 'cam_intrinsic' in cam_data.keys():
                intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2keyego, sensor2sensor = \
                self.get_sensor2ego_transformation(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            sensor2keyego_aug, sensor2sensor_aug, q, sweepsensor2sweepego_tran = \
                self.get_sensor2ego_transformation_aug(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            results['bev_aug'].update({ 'cam'+ str(cam_names.index(cam_name)):
                                       {'sensor2ego_rotation': q}
                                       })
            results['bev_aug']['cam'+ str(cam_names.index(cam_name))].update(
                                       {'sensor2ego_translation': sweepsensor2sweepego_tran}
                                       )
            # print(results['bev_aug'][cam_name])
            # print(results['bev_aug'][cam_name]['sensor2ego_rotation'])
            # print(Quaternion(results['bev_aug'][cam_name]['sensor2ego_rotation']))
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            rot_aug = sensor2keyego_aug[:3, :3]
            tran_aug = sensor2sensor_aug[:3, 3]
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            # if cam_name == 'Camera_Back' and ('cam_intrinsic' not in cam_data.keys()):
            #     resize = resize * np.random.uniform(low=1.0, high=1.5)
            if self.is_train and ('Target_Focal' in self.data_config.keys()):
                H, W = img.height, img.width
                fH, fW = self.data_config['input_size']
                resize_11_w = np.array(float(fW) / float(W))
                resize_11_H = np.array(float(fH) / float(H))
                # 保证不要太多黑边
                resize_11_w_low = resize_11_w * 0.94
                resize_11_H_low = resize_11_H * 0.94
                resize_11_w_high = resize_11_w * 1.4
                resize_11_H_high = resize_11_H * 1.4
                # resize_intri_low = self.data_config['Target_Focal_low'] / np.array(intrin[0, 0])
                # resize_intri_high = self.data_config['Target_Focal_hight'] / np.array(intrin[0, 0])
                resize_intri_target = self.data_config['Target_Focal'] / np.array(intrin[0, 0])
                resize_low = np.max([resize_11_w_low, resize_11_H_low])
                resize_height = np.min([resize_11_w_high, resize_11_H_high])
                np.random.choice([0, 1])
                if resize_low<resize_intri_target and resize_intri_target<resize_height:
                    resize = np.random.uniform(low=resize_intri_target-0.07, high=resize_intri_target+0.07)
                elif resize_intri_target>=resize_height:
                    if np.random.choice([0, 1]):
                        resize = 0.8 * resize + 0.2 * resize_intri_target
                elif resize_intri_target <= resize_low:
                    if np.random.choice([0, 1]):
                        resize = 0.8 * resize + 0.2 * resize_intri_target

            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)
            # get actual intrins
            intrins_actual = copy.deepcopy(intrin)
            intrins_actual[0, :] = resize * intrins_actual[0, :]
            intrins_actual[1, :] = resize * intrins_actual[1, :]
            intrins_actual[0, 2] = intrins_actual[0, 2] - crop[0]
            intrins_actual[1, 2] = intrins_actual[1, 2] - crop[1]
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            results['img_aug'].update({'cam' + str(cam_names.index(cam_name)): [img_augs, post_rot, post_tran]})
            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))

            intrins.append(intrin)
            intrins_actuals.append(intrins_actual)
            rots.append(rot)
            trans.append(tran)
            rot_augs.append(rot_aug)
            tran_augs.append(tran_aug)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)

        if len(cam_names)==5:
            cam_names = cam_names + ['CAM_Vitual']
            results['cam_names'] = cam_names
            temp = self.normalize_img(img)
            temp = 0.0*temp
            imgs.append(temp)
            results['curr']['cams'].update({'CAM_Vitual': results['curr']['cams'][cam_name]})
            results['img_aug'].update({'cam' + str(5): [img_augs, post_rot, post_tran]})
            results['bev_aug'].update({'cam' + str(5):
                                       {'sensor2ego_rotation': q}
                                       })
            results['bev_aug']['cam'+ str(5)].update(
                                       {'sensor2ego_translation': sweepsensor2sweepego_tran}
                                       )

            intrins.append(intrin*0.000001)
            intrins_actuals.append(intrins_actual*0.000001)
            rots.append(rot)
            trans.append(tran)
            rot_augs.append(rot_aug)
            tran_augs.append(tran_aug)
            post_rots.append(post_rot)
            post_trans.append(post_tran)


        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2keyego, sensor2sensor = \
                        self.get_sensor2ego_transformation(adj_info,
                                                           results['curr'],
                                                           cam_name,
                                                           self.ego_cam)
                    rot = adjsensor2keyego[:3, :3]
                    tran = adjsensor2keyego[:3, 3]
                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)

        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        rot_augs = torch.stack(rot_augs)
        tran_augs = torch.stack(tran_augs)
        results['bev_aug'].update({'tran_augs': tran_augs})
        results['bev_aug'].update({'rot_augs': rot_augs})
        intrins = torch.stack(intrins)
        intrins_actuals = torch.stack(intrins_actuals)
        # print(intrins_actuals)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        sensor2sensors = torch.stack(sensor2sensors)
        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        return (imgs, rots, trans, intrins, post_rots, post_trans, intrins_actuals)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        # print('results[img_aug]', results['img_aug'])
        # print('results[bev_aug]', results['bev_aug'])
        return results




@PIPELINES.register_module()
class PointToMultiViewDepth_Uni(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config
        self.c = torch.sqrt(100.0 / torch.square(torch.tensor(450)) + 100.0 / torch.square(torch.tensor(450)))

    def points2depthmap_vitural(self, points, height, width, act_intrins):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        # 转换为相对深度
        depth = (depth * torch.sqrt(
            100.0 / torch.square(act_intrins[0][0]) + 100.0 / torch.square(act_intrins[1][1]))) / self.c
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                        depth < 100) & (
                        depth >= self.grid_config['depth'][0])
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

    def points2depthmap_real(self, points, height, width, act_intrins):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < 100) & (                 # self.grid_config['depth'][1]
                    depth >= self.grid_config['depth'][0])
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


    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda, act_intrins = results['img_inputs'][4:8]
        depth_map_list = []
        depth_map_list_real = []
        lidar2img_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            if 'lidar2ego_rotation' in results['curr'].keys():
                # print('111111111111111111')
                # print(results['curr']['lidar2ego_rotation'])
                # print('2222222')
                # print(Quaternion(
                #     results['curr']['lidar2ego_rotation']))
                # print('33333')
                # print(Quaternion(
                #     results['curr']['lidar2ego_rotation']).rotation_matrix)
                lidar2lidarego[:3, :3] = Quaternion(
                    results['curr']['lidar2ego_rotation']).rotation_matrix
                lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
                lidar2lidarego = torch.from_numpy(lidar2lidarego)
            else:
                lidar2lidarego = results['curr']['lidar_to_ego_matrix']
                lidar2lidarego[1, :] = -lidar2lidarego[1, :]
                lidar2lidarego = torch.from_numpy(lidar2lidarego).float()

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            act_intrin = act_intrins[cid, :, :]

            act_intri = np.eye(4, dtype=np.float32)
            act_intri = torch.from_numpy(act_intri)
            act_intri[:3, :3] = act_intrins[cid]
            if 'lidar2ego_rotation' in results['curr'].keys():
                lidarego2global = np.eye(4, dtype=np.float32)
                lidarego2global[:3, :3] = Quaternion(
                    results['curr']['ego2global_rotation']).rotation_matrix
                lidarego2global[:3, 3] = results['curr']['ego2global_translation']
                lidarego2global = torch.from_numpy(lidarego2global)

                camego2global = np.eye(4, dtype=np.float32)
                camego2global[:3, :3] = Quaternion(
                    results['curr']['cams'][cam_name]
                    ['ego2global_rotation']).rotation_matrix
                camego2global[:3, 3] = results['curr']['cams'][cam_name][
                    'ego2global_translation']
                camego2global = torch.from_numpy(camego2global)

                lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                    lidarego2global.matmul(lidar2lidarego))
                lidar2img = cam2img.matmul(lidar2cam)
            else:
                lidar2cam = torch.inverse(cam2camego).matmul(lidar2lidarego)
                lidar2img = cam2img.matmul(lidar2cam)


            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map_real = self.points2depthmap_real(points_img, imgs.shape[2],
                                                       imgs.shape[3], act_intrin)
            depth_map_vitural = self.points2depthmap_vitural(points_img, imgs.shape[2],
                                                             imgs.shape[3], act_intrin)
            depth_map_list.append(depth_map_vitural)
            depth_map_list_real.append(depth_map_real)
            # print('lidar2img.detach().numpy()', lidar2img.detach().numpy())

            ida_mat = torch.eye(4)
            # print('post_rots[cid]', post_rots[cid])
            # print('post_trans[cid]', post_trans[cid])
            ida_mat[:2, :2] = post_rots[cid][:2, :2]
            ida_mat[:2, 2] = post_trans[cid][:2]

            if 'bda_mat' in results.keys():
                lidar2img = ida_mat @ act_intri.matmul(lidar2cam) @ results['bda_mat']
            else:
                lidar2img = ida_mat @ act_intri.matmul(lidar2cam)
            lidar2img_list.append(lidar2img.detach().numpy())
        # print('lidar2img_list', lidar2img_list)
        depth_map = torch.stack(depth_map_list)
        depth_map_real = torch.stack(depth_map_list_real)
        results['gt_depth'] = depth_map
        results['gt_depth_real'] = depth_map_real
        results['lidar2img'] = lidar2img_list
        return results



