# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
import copy
import cv2
import numpy as np
import sys
sys.path.append('/mnt/cfs/algorithm/hao.lu/Code/UniBEV/mmdet3d/models')
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from .bevdet import BEVDet
from fastsam import FastSAM, FastSAMPrompt
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import torch.nn as nn

@DETECTORS.register_module()
class Scale_BEV_Large2D(BEVDet):
    def __init__(self, large_2D_model_config, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, img_aug, bev_img_aux, detach_flag=1, **kwargs):
        super(Scale_BEV_Large2D, self).__init__(img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs)
        self.detach_flag = detach_flag
        self.img_aug_cfg = img_aug
        self.large_2D_model_config = large_2D_model_config
        self.bev_img_aux_cfg = bev_img_aux
        self.img_aux = builder.build_neck(self.img_aug_cfg)
        self.bev_img_aux = builder.build_neck(self.bev_img_aux_cfg)
        self.bev_index = [0]
        self.large_2D_model = self.build_sam(large_2D_model_config)

        self.project_2D_model = nn.Sequential(
            nn.Conv2d(large_2D_model_config.input_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True)
        )
        self.large_2D_model.to(next(self.bev_img_aux.parameters()).device)
        self.project_2D_model.to(next(self.bev_img_aux.parameters()).device)

    def build_sam(self, sam_config):
        model = FastSAM(sam_config.pretrained)
        model = model.model.model[0:sam_config.depth]
        return model

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          bev_heatmap=None,
                          pseudo_flag=False,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        loss_bev = self.pts_bbox_head.bev_loss(bev_heatmap)
        losses.update({'loss_bev_heatmap_source': loss_bev})
        return losses

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth_real, depth_vitual, loss_dict)


    def cosim(self, img_feature, large_2D_feature):
        # print('img_feature.size()', img_feature.size())
        # print('large_2D_feature', large_2D_feature.size())
        N, Cam, C, H, W = img_feature.size()
        NCam, C, H, W = large_2D_feature.size()
        feature_cosine_similarity = torch.cosine_similarity(img_feature.reshape(NCam, C, -1), large_2D_feature.reshape(NCam, C, -1), dim=2)
        loss = 1 - feature_cosine_similarity.mean().mean()
        return loss


    def extract_img_feat(self, points, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):

        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intri_actually = img
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans, bda, intri_actually)

        rot_augs = kwargs['bev_aug']['rot_augs']
        tran_augs = kwargs['bev_aug']['tran_augs']
        mlp_input_aug = self.img_view_transformer.get_mlp_input(
            rot_augs, tran_augs, intrins, post_rots, post_trans, bda, intri_actually)

        img_feature = self.image_encoder(imgs)
        with torch.no_grad():
            N, Cam, C, H, W = imgs.size()
            large_2D_feature = self.large_2D_model(imgs.reshape(N*Cam, C, H, W))
        large_2D_feature = self.project_2D_model(large_2D_feature)
        cosim_loss = self.cosim(img_feature, large_2D_feature)


        loss_dict = dict()
        x, depth_real, depth_vitual = self.img_view_transformer(
            [img_feature, rots, trans, intrins, post_rots, post_trans, bda, mlp_input, intri_actually])
        bev_feats_source = self.bev_encoder(x)
        # source 2d data
        loss_heatmaps_source, loss_regess_source, heatmaps_source = self.img_aux(img_feature, **kwargs)
        # source bev data
        BEV_features_source, unused_loss_bev_source = self.img_view_transformer.get_BEV_feats_from_voxel(bev_feats_source)
        PV_features, unused_loss = self.img_view_transformer.get_PV_feats(bev_feats_source)
        PV_features_aug, unused_loss_aug = self.img_view_transformer.get_PV_feats_aug(bev_feats_source, [intrins, post_rots, post_trans, bda, intri_actually], **kwargs)
        loss_pv_heatmaps, loss_pv_regess, pv_heatmaps= self.bev_img_aux(PV_features, mlp_input,'img', depth_flag='real', **kwargs)
        loss_pv_heatmaps_aug, loss_pv_regess_aug, pv_heatmaps_aug = self.bev_img_aux(PV_features_aug, mlp_input_aug,'img_aug', depth_flag='real', **kwargs)
        bev_heatmaps, bev_unused_loss = self.bev_img_aux.get_heatmaps(BEV_features_source)
        # consistency
        loss_dict.update({'loss_heatmaps_source': loss_heatmaps_source})
        loss_dict.update({'loss_regess_source': loss_regess_source})
        loss_dict.update({'loss_pv_heatmaps_source': loss_pv_heatmaps})
        loss_dict.update({'loss_pv_regess_source': loss_pv_regess})
        loss_dict.update({'loss_pv_heatmaps_aug_source': loss_pv_heatmaps_aug})
        loss_dict.update({'loss_pv_regess_aug_source': loss_pv_regess_aug})
        loss_dict.update({'loss_large_2D': cosim_loss})

        loss_dict.update({'loss_unused_xxx': 0.0*(unused_loss_bev_source+unused_loss+unused_loss_aug+bev_unused_loss
                                                )})
        pts_feats = None
        return [bev_feats_source], bev_heatmaps,  pts_feats, depth_real, depth_vitual, loss_dict


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _, _= self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['img_metas'] = img_metas
        return bbox_list  # img_metas #

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):


        img_feats, bev_heatmaps, pts_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(
            points, img=img_inputs, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)

        gt_depth = kwargs['ann_maps_2d']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth_vitual)
        losses = dict(loss_depth_source=loss_depth)
        losses.update(loss_dict)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas, bev_heatmap=bev_heatmaps,
                                            pseudo_flag=False, gt_bboxes_ignore=None)
        losses.update(losses_pts)

        return losses





# 加入mask，增加pro的层数
@DETECTORS.register_module()
class Scale_BEV_Large2D_V2(BEVDet):
    def __init__(self, large_2D_model_config, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, img_aug, bev_img_aux, detach_flag=1, **kwargs):
        super(Scale_BEV_Large2D_V2, self).__init__(img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs)
        self.detach_flag = detach_flag
        self.img_aug_cfg = img_aug
        self.large_2D_model_config = large_2D_model_config
        self.bev_img_aux_cfg = bev_img_aux
        self.img_aux = builder.build_neck(self.img_aug_cfg)
        self.bev_img_aux = builder.build_neck(self.bev_img_aux_cfg)
        self.bev_index = [0]
        self.large_2D_model = self.build_sam(large_2D_model_config)

        self.project_2D_model = nn.Sequential(
            nn.Conv2d(large_2D_model_config.input_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True)
        )
        self.large_2D_model.to(next(self.bev_img_aux.parameters()).device)
        self.project_2D_model.to(next(self.bev_img_aux.parameters()).device)

    def build_sam(self, sam_config):
        model = FastSAM(sam_config.pretrained)
        model = model.model.model[0:sam_config.depth]
        return model

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          bev_heatmap=None,
                          pseudo_flag=False,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        loss_bev = self.pts_bbox_head.bev_loss(bev_heatmap)
        losses.update({'loss_bev_heatmap_source': loss_bev})
        return losses

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth_real, depth_vitual, loss_dict)


    def cosim(self, img_feature, large_2D_feature):
        # print('img_feature.size()', img_feature.size())
        # print('large_2D_feature', large_2D_feature.size())
        N, Cam, C, H, W = img_feature.size()
        NCam, C, H, W = large_2D_feature.size()
        feature_cosine_similarity = torch.cosine_similarity(img_feature.reshape(NCam, C, -1), large_2D_feature.reshape(NCam, C, -1), dim=2)
        loss = 1 - feature_cosine_similarity.mean().mean()
        return loss

    def feature_distillation(self, img_feature, large_2D_feature, **kwargs):
        heatmap_masks_2d = kwargs['heatmap_masks_2d']
        N, Cam, C, H, W = img_feature.size()
        img_feature = img_feature[:, :, 0:self.large_2D_model_config.output_chanel, :, :]
        large_2D_feature = large_2D_feature.reshape(N, Cam, self.large_2D_model_config.output_chanel, H, W)

        if type(heatmap_masks_2d) in [list, tuple]:
            heatmap_masks_2d = torch.stack(heatmap_masks_2d)  # ([6, 6, 96, 176, 3])
        heatmap_masks_2d = torch.sum(heatmap_masks_2d, dim=-1) # ([6, 6, 96, 176])
        heatmap_masks_2d = torch.clamp(heatmap_masks_2d, 0, 1)
        heatmap_masks_2d = F.interpolate(heatmap_masks_2d, scale_factor=0.5, mode='bilinear')
        heatmap_masks_2d = heatmap_masks_2d.unsqueeze(2)
        img_feature = torch.mul(heatmap_masks_2d, img_feature)
        large_2D_feature = torch.mul(heatmap_masks_2d, large_2D_feature)
        heatmap_masks_2d = heatmap_masks_2d.mean(0).mean(0).reshape(-1)
        mask = heatmap_masks_2d.ge(0)
        img_feature = img_feature.reshape(N * Cam * self.large_2D_model_config.output_chanel, -1)
        large_2D_feature = large_2D_feature.reshape(N * Cam * self.large_2D_model_config.output_chanel, -1)
        feature_cosine_similarity = torch.cosine_similarity(img_feature[:, mask], large_2D_feature[:, mask], dim=-1)
        loss = 1 - feature_cosine_similarity.mean().mean()
        return loss


    def extract_img_feat(self, points, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):

        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intri_actually = img
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans, bda, intri_actually)

        rot_augs = kwargs['bev_aug']['rot_augs']
        tran_augs = kwargs['bev_aug']['tran_augs']
        mlp_input_aug = self.img_view_transformer.get_mlp_input(
            rot_augs, tran_augs, intrins, post_rots, post_trans, bda, intri_actually)

        img_feature = self.image_encoder(imgs)
        with torch.no_grad():
            N, Cam, C, H, W = imgs.size()
            large_2D_feature = self.large_2D_model(imgs.reshape(N*Cam, C, H, W))
        large_2D_feature = self.project_2D_model(large_2D_feature)
        cosim_loss = self.feature_distillation(img_feature, large_2D_feature, **kwargs)


        loss_dict = dict()
        x, depth_real, depth_vitual = self.img_view_transformer(
            [img_feature, rots, trans, intrins, post_rots, post_trans, bda, mlp_input, intri_actually])
        bev_feats_source = self.bev_encoder(x)
        # source 2d data
        loss_heatmaps_source, loss_regess_source, heatmaps_source = self.img_aux(img_feature, **kwargs)
        # source bev data
        BEV_features_source, unused_loss_bev_source = self.img_view_transformer.get_BEV_feats_from_voxel(bev_feats_source)
        PV_features, unused_loss = self.img_view_transformer.get_PV_feats(bev_feats_source)
        PV_features_aug, unused_loss_aug = self.img_view_transformer.get_PV_feats_aug(bev_feats_source, [intrins, post_rots, post_trans, bda, intri_actually], **kwargs)
        loss_pv_heatmaps, loss_pv_regess, pv_heatmaps= self.bev_img_aux(PV_features, mlp_input, 'img', depth_flag='real', **kwargs)
        loss_pv_heatmaps_aug, loss_pv_regess_aug, pv_heatmaps_aug = self.bev_img_aux(PV_features_aug, mlp_input_aug,'img_aug', depth_flag='real', **kwargs)
        bev_heatmaps, bev_unused_loss = self.bev_img_aux.get_heatmaps(BEV_features_source)
        # consistency
        loss_dict.update({'loss_heatmaps_source': loss_heatmaps_source})
        loss_dict.update({'loss_regess_source': loss_regess_source})
        loss_dict.update({'loss_pv_heatmaps_source': loss_pv_heatmaps})
        loss_dict.update({'loss_pv_regess_source': loss_pv_regess})
        loss_dict.update({'loss_pv_heatmaps_aug_source': loss_pv_heatmaps_aug})
        loss_dict.update({'loss_pv_regess_aug_source': loss_pv_regess_aug})
        loss_dict.update({'loss_large_2D': cosim_loss})

        loss_dict.update({'loss_unused_xxx': 0.0*(unused_loss_bev_source+unused_loss+unused_loss_aug+bev_unused_loss
                                                )})
        pts_feats = None
        return [bev_feats_source], bev_heatmaps,  pts_feats, depth_real, depth_vitual, loss_dict


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _, _= self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['img_metas'] = img_metas
        return bbox_list  # img_metas #

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):


        img_feats, bev_heatmaps, pts_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(
            points, img=img_inputs, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)

        gt_depth = kwargs['ann_maps_2d']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth_vitual)
        losses = dict(loss_depth_source=loss_depth)
        losses.update(loss_dict)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas, bev_heatmap=bev_heatmaps,
                                            pseudo_flag=False, gt_bboxes_ignore=None)
        losses.update(losses_pts)

        return losses





# 加入mask，增加pro的层数，加入多个img_encoder
@DETECTORS.register_module()
class Scale_BEV_Large2D_V3(BEVDet):
    def __init__(self, aux_config, large_2D_model_config, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, img_aug, bev_img_aux, detach_flag=1, **kwargs):
        super(Scale_BEV_Large2D_V3, self).__init__(img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs)
        self.aux_config = aux_config
        self.detach_flag = detach_flag
        self.img_aug_cfg = img_aug
        self.large_2D_model_config = large_2D_model_config
        self.bev_img_aux_cfg = bev_img_aux
        self.img_aux = builder.build_neck(self.img_aug_cfg)
        self.bev_img_aux = builder.build_neck(self.bev_img_aux_cfg)
        self.bev_index = [0]
        self.large_2D_model = self.build_sam(large_2D_model_config)
        self.MSELoss = nn.MSELoss()

        self.project_2D_model = nn.Sequential(
            nn.Conv2d(large_2D_model_config.input_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True)
        )
        self.large_2D_model.to(next(self.bev_img_aux.parameters()).device)
        self.project_2D_model.to(next(self.bev_img_aux.parameters()).device)

        self.img_backbone_aux0 = builder.build_backbone(self.img_backbone_config).to(next(self.bev_img_aux.parameters()).device)
        self.img_neck_aux0 = builder.build_neck(self.img_neck_config).to(next(self.bev_img_aux.parameters()).device)
        self.img_backbone_aux1 = builder.build_backbone(self.img_backbone_config).to(next(self.bev_img_aux.parameters()).device)
        self.img_neck_aux1 = builder.build_neck(self.img_neck_config).to(next(self.bev_img_aux.parameters()).device)

    def image_encoder_aux0(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone_aux0(imgs)

        if self.with_img_neck:
            x = self.img_neck_aux0(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def image_encoder_aux1(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone_aux1(imgs)

        if self.with_img_neck:
            x = self.img_neck_aux1(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def build_sam(self, sam_config):
        model = FastSAM(sam_config.pretrained)
        model = model.model.model[0:sam_config.depth]
        return model

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          bev_heatmap=None,
                          pseudo_flag=False,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        loss_bev = self.pts_bbox_head.bev_loss(bev_heatmap)
        losses.update({'loss_bev_heatmap_source': loss_bev})
        return losses

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth_real, depth_vitual, loss_dict)


    def cosim(self, img_feature, large_2D_feature):
        # print('img_feature.size()', img_feature.size())
        # print('large_2D_feature', large_2D_feature.size())
        N, Cam, C, H, W = img_feature.size()
        NCam, C, H, W = large_2D_feature.size()
        feature_cosine_similarity = torch.cosine_similarity(img_feature.reshape(NCam, C, -1), large_2D_feature.reshape(NCam, C, -1), dim=2)
        loss = 1 - feature_cosine_similarity.mean().mean()
        return loss

    def feature_distillation(self, img_feature, large_2D_feature, **kwargs):
        heatmap_masks_2d = kwargs['heatmap_masks_2d']
        N, Cam, C, H, W = img_feature.size()
        img_feature = img_feature[:, :, 0:self.large_2D_model_config.output_chanel, :, :]
        large_2D_feature = large_2D_feature.reshape(N, Cam, self.large_2D_model_config.output_chanel, H, W)

        if type(heatmap_masks_2d) in [list, tuple]:
            heatmap_masks_2d = torch.stack(heatmap_masks_2d)  # ([6, 6, 96, 176, 3])
        heatmap_masks_2d = torch.sum(heatmap_masks_2d, dim=-1) # ([6, 6, 96, 176])
        heatmap_masks_2d = torch.clamp(heatmap_masks_2d, 0, 1)
        heatmap_masks_2d = F.interpolate(heatmap_masks_2d, scale_factor=0.5, mode='bilinear')
        heatmap_masks_2d = heatmap_masks_2d.unsqueeze(2)
        img_feature = torch.mul(heatmap_masks_2d, img_feature)
        large_2D_feature = torch.mul(heatmap_masks_2d, large_2D_feature)
        heatmap_masks_2d = heatmap_masks_2d.mean(0).mean(0).reshape(-1)
        mask = heatmap_masks_2d.ge(0)
        img_feature = img_feature.reshape(N * Cam * self.large_2D_model_config.output_chanel, -1)
        large_2D_feature = large_2D_feature.reshape(N * Cam * self.large_2D_model_config.output_chanel, -1)
        feature_cosine_similarity = torch.cosine_similarity(img_feature[:, mask], large_2D_feature[:, mask], dim=-1)
        loss = 1 - feature_cosine_similarity.mean().mean()
        return loss

    def feature_distillation_L1(self, img_feature, img_feature_biased, weight):
        img_featuredetach = img_feature.detach()
        diff = torch.pow(img_featuredetach - img_feature_biased, 2)
        diff = diff.mean(-1).mean(-1).mean(-1).mean(-1)
        loss = diff * weight
        return 5*loss.mean()



    def weight(self, GeoMetric):
        if type(GeoMetric) in [list, tuple]:
            GeoMetric = torch.stack(GeoMetric)
        weight0 = torch.exp(GeoMetric / 400).clamp(0, 3)
        weight1 = torch.exp((400-GeoMetric) / 400)
        return weight0, weight1

    def extract_img_feat(self, points, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):

        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intri_actually = img
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans, bda, intri_actually)

        rot_augs = kwargs['bev_aug']['rot_augs']
        tran_augs = kwargs['bev_aug']['tran_augs']
        mlp_input_aug = self.img_view_transformer.get_mlp_input(
            rot_augs, tran_augs, intrins, post_rots, post_trans, bda, intri_actually)
        alpha = 0.9
        with torch.no_grad():
            for param1, param2 in zip(self.img_backbone_aux0.parameters(), self.image_encoder.parameters()):
                smoothed_param = alpha * param1.detach() + (1 - alpha) * param2.detach()
                self.image_encoder_aux0.copy_(smoothed_param)
            for param1, param2 in zip(self.image_encoder_aux1.parameters(), self.image_encoder.parameters()):
                smoothed_param = alpha * param1.detach() + (1 - alpha) * param2.detach()
                self.image_encoder_aux1.copy_(smoothed_param)

        img_feature = self.image_encoder(imgs)
        img_feature_0 = self.image_encoder_aux0(imgs)
        img_feature_1 = self.image_encoder_aux1(imgs)
        weight0, weight1 = self.weight(kwargs['GeoMetric'])
        loss_biased_0 = self.feature_distillation_L1(img_feature, img_feature_0, weight0)
        loss_biased_1 = self.feature_distillation_L1(img_feature, img_feature_1, weight1)

        if random.random() < 0.1:
            N = img_feature.shape[0]
            random_index = random.randint(0, N-1)
            random_Plus = random.randint(0, 2-1)
            img_feature = torch.cat([img_feature, img_feature_0, img_feature_1], dim=0)
            old_index = torch.arange(N*3)
            old_index[random_index] = random_Plus*N+random_index
            img_feature = img_feature[old_index[0:N]]



        with torch.no_grad():
            N, Cam, C, H, W = imgs.size()
            large_2D_feature = self.large_2D_model(imgs.reshape(N*Cam, C, H, W))
        large_2D_feature = self.project_2D_model(large_2D_feature)
        cosim_loss = self.feature_distillation(img_feature, large_2D_feature, **kwargs)


        loss_dict = dict()
        x, depth_real, depth_vitual = self.img_view_transformer(
            [img_feature, rots, trans, intrins, post_rots, post_trans, bda, mlp_input, intri_actually])


        bev_feats_source = self.bev_encoder(x)
        # source 2d data
        loss_heatmaps_source, loss_regess_source, heatmaps_source = self.img_aux(img_feature, **kwargs)
        # source bev data
        BEV_features_source, unused_loss_bev_source = self.img_view_transformer.get_BEV_feats_from_voxel(bev_feats_source)
        PV_features, unused_loss = self.img_view_transformer.get_PV_feats(bev_feats_source)
        PV_features_aug, unused_loss_aug = self.img_view_transformer.get_PV_feats_aug(bev_feats_source, [intrins, post_rots, post_trans, bda, intri_actually], **kwargs)
        loss_pv_heatmaps, loss_pv_regess, pv_heatmaps= self.bev_img_aux(PV_features, mlp_input, 'img', depth_flag='real', **kwargs)
        loss_pv_heatmaps_aug, loss_pv_regess_aug, pv_heatmaps_aug = self.bev_img_aux(PV_features_aug, mlp_input_aug,'img_aug', depth_flag='real', **kwargs)
        bev_heatmaps, bev_unused_loss = self.bev_img_aux.get_heatmaps(BEV_features_source)
        # consistency
        loss_dict.update({'loss_heatmaps_source': loss_heatmaps_source})
        loss_dict.update({'loss_regess_source': loss_regess_source})
        loss_dict.update({'loss_pv_heatmaps_source': loss_pv_heatmaps})
        loss_dict.update({'loss_pv_regess_source': loss_pv_regess})
        loss_dict.update({'loss_pv_heatmaps_aug_source': loss_pv_heatmaps_aug})
        loss_dict.update({'loss_pv_regess_aug_source': loss_pv_regess_aug})
        loss_dict.update({'loss_large_2D': cosim_loss})
        loss_dict.update({'loss_biased': loss_biased_0+loss_biased_1})

        loss_dict.update({'loss_unused_xxx': 0.0*(unused_loss_bev_source+unused_loss+unused_loss_aug+bev_unused_loss
                                                )})
        pts_feats = None
        return [bev_feats_source], bev_heatmaps,  pts_feats, depth_real, depth_vitual, loss_dict


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _, _= self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['img_metas'] = img_metas
        return bbox_list  # img_metas #

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):


        img_feats, bev_heatmaps, pts_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(
            points, img=img_inputs, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)

        gt_depth = kwargs['ann_maps_2d']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth_vitual)
        losses = dict(loss_depth_source=loss_depth)
        losses.update(loss_dict)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas, bev_heatmap=bev_heatmaps,
                                            pseudo_flag=False, gt_bboxes_ignore=None)
        losses.update(losses_pts)

        return losses





# 加入mask，增加pro的层数，加入多个img_encoder,加入EMA
@DETECTORS.register_module()
class Scale_BEV_Large2D_V4(BEVDet):
    def __init__(self, aux_config, large_2D_model_config, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, img_aug, bev_img_aux, detach_flag=1, **kwargs):
        super(Scale_BEV_Large2D_V4, self).__init__(img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs)
        self.aux_config = aux_config
        self.detach_flag = detach_flag
        self.img_aug_cfg = img_aug
        self.large_2D_model_config = large_2D_model_config
        self.bev_img_aux_cfg = bev_img_aux
        self.img_aux = builder.build_neck(self.img_aug_cfg)
        self.bev_img_aux = builder.build_neck(self.bev_img_aux_cfg)
        self.bev_index = [0]
        self.large_2D_model = self.build_sam(large_2D_model_config)
        self.MSELoss = nn.MSELoss()

        self.project_2D_model = nn.Sequential(
            nn.Conv2d(large_2D_model_config.input_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True)
        )
        self.large_2D_model.to(next(self.bev_img_aux.parameters()).device)
        self.project_2D_model.to(next(self.bev_img_aux.parameters()).device)

        self.img_backbone_aux0 = builder.build_backbone(self.img_backbone_config).to(next(self.bev_img_aux.parameters()).device)
        self.img_neck_aux0 = builder.build_neck(self.img_neck_config).to(next(self.bev_img_aux.parameters()).device)
        self.img_backbone_aux1 = builder.build_backbone(self.img_backbone_config).to(next(self.bev_img_aux.parameters()).device)
        self.img_neck_aux1 = builder.build_neck(self.img_neck_config).to(next(self.bev_img_aux.parameters()).device)

    def image_encoder_aux0(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone_aux0(imgs)

        if self.with_img_neck:
            x = self.img_neck_aux0(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def image_encoder_aux1(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone_aux1(imgs)

        if self.with_img_neck:
            x = self.img_neck_aux1(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def build_sam(self, sam_config):
        model = FastSAM(sam_config.pretrained)
        model = model.model.model[0:sam_config.depth]
        return model

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          bev_heatmap=None,
                          pseudo_flag=False,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        loss_bev = self.pts_bbox_head.bev_loss(bev_heatmap)
        losses.update({'loss_bev_heatmap_source': loss_bev})
        return losses

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth_real, depth_vitual, loss_dict)


    def cosim(self, img_feature, large_2D_feature):
        # print('img_feature.size()', img_feature.size())
        # print('large_2D_feature', large_2D_feature.size())
        N, Cam, C, H, W = img_feature.size()
        NCam, C, H, W = large_2D_feature.size()
        feature_cosine_similarity = torch.cosine_similarity(img_feature.reshape(NCam, C, -1), large_2D_feature.reshape(NCam, C, -1), dim=2)
        loss = 1 - feature_cosine_similarity.mean().mean()
        return loss

    def feature_distillation(self, img_feature, large_2D_feature, **kwargs):
        heatmap_masks_2d = kwargs['heatmap_masks_2d']
        N, Cam, C, H, W = img_feature.size()
        img_feature = img_feature[:, :, 0:self.large_2D_model_config.output_chanel, :, :]
        large_2D_feature = large_2D_feature.reshape(N, Cam, self.large_2D_model_config.output_chanel, H, W)

        if type(heatmap_masks_2d) in [list, tuple]:
            heatmap_masks_2d = torch.stack(heatmap_masks_2d)  # ([6, 6, 96, 176, 3])
        heatmap_masks_2d = torch.sum(heatmap_masks_2d, dim=-1) # ([6, 6, 96, 176])
        heatmap_masks_2d = torch.clamp(heatmap_masks_2d, 0, 1)
        heatmap_masks_2d = F.interpolate(heatmap_masks_2d, scale_factor=0.5, mode='bilinear')
        heatmap_masks_2d = heatmap_masks_2d.unsqueeze(2)
        img_feature = torch.mul(heatmap_masks_2d, img_feature)
        large_2D_feature = torch.mul(heatmap_masks_2d, large_2D_feature)
        heatmap_masks_2d = heatmap_masks_2d.mean(0).mean(0).reshape(-1)
        mask = heatmap_masks_2d.ge(0)
        img_feature = img_feature.reshape(N * Cam * self.large_2D_model_config.output_chanel, -1)
        large_2D_feature = large_2D_feature.reshape(N * Cam * self.large_2D_model_config.output_chanel, -1)
        feature_cosine_similarity = torch.cosine_similarity(img_feature[:, mask], large_2D_feature[:, mask], dim=-1)
        loss = 1 - feature_cosine_similarity.mean().mean()
        return loss

    def feature_distillation_L1(self, img_feature, img_feature_biased, weight):
        img_featuredetach = img_feature.detach()
        diff = torch.pow(img_featuredetach - img_feature_biased, 2)
        diff = diff.mean(-1).mean(-1).mean(-1).mean(-1)
        loss = diff * weight
        return loss.mean()





    def weight(self, GeoMetric):
        if type(GeoMetric) in [list, tuple]:
            GeoMetric = torch.stack(GeoMetric)
        weight0 = torch.exp(GeoMetric / 400).clamp(0, 3)
        weight1 = torch.exp((400-GeoMetric) / 400).clamp(0, 3)
        return weight0, weight1

    def extract_img_feat(self, points, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):

        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intri_actually = img
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans, bda, intri_actually)

        rot_augs = kwargs['bev_aug']['rot_augs']
        tran_augs = kwargs['bev_aug']['tran_augs']
        mlp_input_aug = self.img_view_transformer.get_mlp_input(
            rot_augs, tran_augs, intrins, post_rots, post_trans, bda, intri_actually)

        alpha = 0.1
        with torch.no_grad():
            for name, param in self.img_backbone.named_parameters():
                param = param.detach()
                self.img_backbone_aux0.state_dict()[name] = alpha * self.img_backbone_aux0.state_dict()[name].detach() + (1 - alpha)*param.detach()
                self.img_backbone_aux1.state_dict()[name] = alpha * self.img_backbone_aux1.state_dict()[name].detach() + (1 - alpha)*param.detach()
            for name, param in self.img_neck.named_parameters():
                param = param.detach()
                self.img_neck_aux0.state_dict()[name] = alpha * self.img_neck_aux0.state_dict()[name].detach() + (1 - alpha)*param.detach()
                self.img_neck_aux1.state_dict()[name] = alpha * self.img_neck_aux1.state_dict()[name].detach() + (1 - alpha)*param.detach()
            # for param1, param2 in zip(self.img_backbone_aux0.parameters(), self.img_backbone.parameters()):
            #     smoothed_param = alpha * param1.detach() + (1 - alpha) * param2.detach()
            #     self.img_backbone_aux0.copy_(smoothed_param)
            # for param1, param2 in zip(self.img_neck_aux0.parameters(), self.img_neck.parameters()):
            #     smoothed_param = alpha * param1.detach() + (1 - alpha) * param2.detach()
            #     self.img_neck_aux0.copy_(smoothed_param)
            #
            # for param1, param2 in zip(self.img_backbone_aux1.parameters(), self.img_backbone.parameters()):
            #     smoothed_param = alpha * param1.detach() + (1 - alpha) * param2.detach()
            #     self.img_backbone_aux1.copy_(smoothed_param)
            # for param1, param2 in zip(self.img_neck_aux1.parameters(), self.img_neck.parameters()):
            #     smoothed_param = alpha * param1.detach() + (1 - alpha) * param2.detach()
            #     self.img_neck_aux1.copy_(smoothed_param)


        img_feature = self.image_encoder(imgs)
        img_feature_0 = self.image_encoder_aux0(imgs)
        img_feature_1 = self.image_encoder_aux1(imgs)
        weight0, weight1 = self.weight(kwargs['GeoMetric'])
        loss_biased_0 = self.feature_distillation_L1(img_feature, img_feature_0, weight0)
        loss_biased_1 = self.feature_distillation_L1(img_feature, img_feature_1, weight1)

        if random.random() < 0.1:
            N = img_feature.shape[0]
            random_index = random.randint(0, N-1)
            random_Plus = random.randint(0, 2-1)
            img_feature = torch.cat([img_feature, img_feature_0, img_feature_1], dim=0)
            old_index = torch.arange(N*3)
            old_index[random_index] = random_Plus*N+random_index
            img_feature = img_feature[old_index[0:N]]


        with torch.no_grad():
            N, Cam, C, H, W = imgs.size()
            large_2D_feature = self.large_2D_model(imgs.reshape(N*Cam, C, H, W))
        large_2D_feature = self.project_2D_model(large_2D_feature)
        cosim_loss = self.feature_distillation(img_feature, large_2D_feature, **kwargs)


        loss_dict = dict()
        x, depth_real, depth_vitual = self.img_view_transformer(
            [img_feature, rots, trans, intrins, post_rots, post_trans, bda, mlp_input, intri_actually])


        bev_feats_source = self.bev_encoder(x)
        # source 2d data
        loss_heatmaps_source, loss_regess_source, heatmaps_source = self.img_aux(img_feature, **kwargs)
        # source bev data
        BEV_features_source, unused_loss_bev_source = self.img_view_transformer.get_BEV_feats_from_voxel(bev_feats_source)
        PV_features, unused_loss = self.img_view_transformer.get_PV_feats(bev_feats_source)
        PV_features_aug, unused_loss_aug = self.img_view_transformer.get_PV_feats_aug(bev_feats_source, [intrins, post_rots, post_trans, bda, intri_actually], **kwargs)
        loss_pv_heatmaps, loss_pv_regess, pv_heatmaps= self.bev_img_aux(PV_features, mlp_input, 'img', depth_flag='real', **kwargs)
        loss_pv_heatmaps_aug, loss_pv_regess_aug, pv_heatmaps_aug = self.bev_img_aux(PV_features_aug, mlp_input_aug,'img_aug', depth_flag='real', **kwargs)
        bev_heatmaps, bev_unused_loss = self.bev_img_aux.get_heatmaps(BEV_features_source)
        # consistency
        loss_dict.update({'loss_heatmaps_source': loss_heatmaps_source})
        loss_dict.update({'loss_regess_source': loss_regess_source})
        loss_dict.update({'loss_pv_heatmaps_source': loss_pv_heatmaps})
        loss_dict.update({'loss_pv_regess_source': loss_pv_regess})
        loss_dict.update({'loss_pv_heatmaps_aug_source': loss_pv_heatmaps_aug})
        loss_dict.update({'loss_pv_regess_aug_source': loss_pv_regess_aug})
        loss_dict.update({'loss_large_2D': cosim_loss})
        loss_dict.update({'loss_biased': loss_biased_0+loss_biased_1})

        loss_dict.update({'loss_unused_xxx': 0.0*(unused_loss_bev_source+unused_loss+unused_loss_aug+bev_unused_loss
                                                )})
        pts_feats = None
        return [bev_feats_source], bev_heatmaps,  pts_feats, depth_real, depth_vitual, loss_dict


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _, _= self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['img_metas'] = img_metas
        return bbox_list  # img_metas #

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):


        img_feats, bev_heatmaps, pts_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(
            points, img=img_inputs, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)

        gt_depth = kwargs['ann_maps_2d']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth_vitual)
        losses = dict(loss_depth_source=loss_depth)
        losses.update(loss_dict)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas, bev_heatmap=bev_heatmaps,
                                            pseudo_flag=False, gt_bboxes_ignore=None)
        losses.update(losses_pts)

        return losses





# 加入mask，增加pro的层数，加入多个img_encoder,加入EMA
# 加入消融
@DETECTORS.register_module()
class Scale_BEV_Large2D_V5(BEVDet):
    def __init__(self, aux_config, loss_weight, large_2D_model_config, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, img_aug, bev_img_aux, detach_flag=1, **kwargs):
        super(Scale_BEV_Large2D_V5, self).__init__(img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs)
        self.loss_weight = loss_weight
        self.aux_config = aux_config
        self.detach_flag = detach_flag
        self.img_aug_cfg = img_aug
        self.large_2D_model_config = large_2D_model_config
        self.bev_img_aux_cfg = bev_img_aux
        self.img_aux = builder.build_neck(self.img_aug_cfg)
        self.bev_img_aux = builder.build_neck(self.bev_img_aux_cfg)
        self.bev_index = [0]
        self.large_2D_model = self.build_sam(large_2D_model_config)
        self.MSELoss = nn.MSELoss()

        self.project_2D_model = nn.Sequential(
            nn.Conv2d(large_2D_model_config.input_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(large_2D_model_config.output_chanel, large_2D_model_config.output_chanel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(large_2D_model_config.output_chanel),
            nn.ReLU(inplace=True)
        )
        self.large_2D_model.to(next(self.bev_img_aux.parameters()).device)
        self.project_2D_model.to(next(self.bev_img_aux.parameters()).device)

        self.img_backbone_aux0 = builder.build_backbone(self.img_backbone_config).to(next(self.bev_img_aux.parameters()).device)
        self.img_neck_aux0 = builder.build_neck(self.img_neck_config).to(next(self.bev_img_aux.parameters()).device)
        self.img_backbone_aux1 = builder.build_backbone(self.img_backbone_config).to(next(self.bev_img_aux.parameters()).device)
        self.img_neck_aux1 = builder.build_neck(self.img_neck_config).to(next(self.bev_img_aux.parameters()).device)

    def image_encoder_aux0(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone_aux0(imgs)

        if self.with_img_neck:
            x = self.img_neck_aux0(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def image_encoder_aux1(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone_aux1(imgs)

        if self.with_img_neck:
            x = self.img_neck_aux1(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def build_sam(self, sam_config):
        model = FastSAM(sam_config.pretrained)
        model = model.model.model[0:sam_config.depth]
        return model

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          bev_heatmap=None,
                          pseudo_flag=False,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        loss_bev = self.pts_bbox_head.bev_loss(bev_heatmap)
        losses.update({'loss_bev_heatmap_source': loss_bev})
        return losses

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth_real, depth_vitual, loss_dict)


    def cosim(self, img_feature, large_2D_feature):
        # print('img_feature.size()', img_feature.size())
        # print('large_2D_feature', large_2D_feature.size())
        N, Cam, C, H, W = img_feature.size()
        NCam, C, H, W = large_2D_feature.size()
        feature_cosine_similarity = torch.cosine_similarity(img_feature.reshape(NCam, C, -1), large_2D_feature.reshape(NCam, C, -1), dim=2)
        loss = 1 - feature_cosine_similarity.mean().mean()
        return loss

    def feature_distillation(self, img_feature, large_2D_feature, **kwargs):
        heatmap_masks_2d = kwargs['heatmap_masks_2d']
        N, Cam, C, H, W = img_feature.size()
        img_feature = img_feature[:, :, 0:self.large_2D_model_config.output_chanel, :, :]
        large_2D_feature = large_2D_feature.reshape(N, Cam, self.large_2D_model_config.output_chanel, H, W)

        if type(heatmap_masks_2d) in [list, tuple]:
            heatmap_masks_2d = torch.stack(heatmap_masks_2d)  # ([6, 6, 96, 176, 3])
        heatmap_masks_2d = torch.sum(heatmap_masks_2d, dim=-1) # ([6, 6, 96, 176])
        heatmap_masks_2d = torch.clamp(heatmap_masks_2d, 0, 1)
        heatmap_masks_2d = F.interpolate(heatmap_masks_2d, scale_factor=0.5, mode='bilinear')
        heatmap_masks_2d = heatmap_masks_2d.unsqueeze(2)
        img_feature = torch.mul(heatmap_masks_2d, img_feature)
        large_2D_feature = torch.mul(heatmap_masks_2d, large_2D_feature)
        heatmap_masks_2d = heatmap_masks_2d.mean(0).mean(0).reshape(-1)
        mask = heatmap_masks_2d.ge(0)
        img_feature = img_feature.reshape(N * Cam * self.large_2D_model_config.output_chanel, -1)
        large_2D_feature = large_2D_feature.reshape(N * Cam * self.large_2D_model_config.output_chanel, -1)
        feature_cosine_similarity = torch.cosine_similarity(img_feature[:, mask], large_2D_feature[:, mask], dim=-1)
        loss = 1 - feature_cosine_similarity.mean().mean()
        return loss

    def feature_distillation_L1(self, img_feature, img_feature_biased, weight):
        img_featuredetach = img_feature.detach()
        diff = torch.pow(img_featuredetach - img_feature_biased, 2)
        diff = diff.mean(-1).mean(-1).mean(-1).mean(-1)
        loss = diff * weight
        return loss.mean()

    def feature_distillation_cos(self, img_feature, img_feature_biased, weight):
        N, Cam, C, H, W = img_feature.size()
        img_feature = img_feature.reshape(N, Cam * C, -1)
        img_feature_biased = img_feature_biased.reshape(N, Cam * C, -1)
        feature_cosine_similarity = torch.cosine_similarity(img_feature, img_feature_biased, dim=-1)
        loss = 1 - feature_cosine_similarity.mean(-1)
        loss = loss * weight
        return loss.mean()

    def weight(self, GeoMetric):
        if type(GeoMetric) in [list, tuple]:
            GeoMetric = torch.stack(GeoMetric)
        weight0 = torch.exp(GeoMetric / 400).clamp(0, 3)
        weight1 = torch.exp((400-GeoMetric) / 400).clamp(0, 3)
        return weight0, weight1

    def extract_img_feat(self, points, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):

        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intri_actually = img
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans, bda, intri_actually)

        rot_augs = kwargs['bev_aug']['rot_augs']
        tran_augs = kwargs['bev_aug']['tran_augs']
        mlp_input_aug = self.img_view_transformer.get_mlp_input(
            rot_augs, tran_augs, intrins, post_rots, post_trans, bda, intri_actually)

        alpha = self.aux_config.alpha
        with torch.no_grad():
            for name, param in self.img_backbone.named_parameters():
                param = param.detach()
                self.img_backbone_aux0.state_dict()[name] = alpha * self.img_backbone_aux0.state_dict()[name].detach() + (1 - alpha)*param.detach()
                self.img_backbone_aux1.state_dict()[name] = alpha * self.img_backbone_aux1.state_dict()[name].detach() + (1 - alpha)*param.detach()
            for name, param in self.img_neck.named_parameters():
                param = param.detach()
                self.img_neck_aux0.state_dict()[name] = alpha * self.img_neck_aux0.state_dict()[name].detach() + (1 - alpha)*param.detach()
                self.img_neck_aux1.state_dict()[name] = alpha * self.img_neck_aux1.state_dict()[name].detach() + (1 - alpha)*param.detach()


        img_feature = self.image_encoder(imgs)
        img_feature_0 = self.image_encoder_aux0(imgs)
        img_feature_1 = self.image_encoder_aux1(imgs)
        weight0, weight1 = self.weight(kwargs['GeoMetric'])
        loss_biased_0 = self.feature_distillation_cos(img_feature, img_feature_0, weight0)
        loss_biased_1 = self.feature_distillation_cos(img_feature, img_feature_1, weight1)

        if random.random() < self.aux_config.ratio:
            N = img_feature.shape[0]
            random_index = random.randint(0, N-1)
            random_Plus = random.randint(0, 2-1)
            img_feature = torch.cat([img_feature, img_feature_0, img_feature_1], dim=0)
            old_index = torch.arange(N*3)
            old_index[random_index] = random_Plus*N+random_index
            img_feature = img_feature[old_index[0:N]]


        with torch.no_grad():
            N, Cam, C, H, W = imgs.size()
            large_2D_feature = self.large_2D_model(imgs.reshape(N*Cam, C, H, W))
        large_2D_feature = self.project_2D_model(large_2D_feature)
        cosim_loss = self.feature_distillation(img_feature, large_2D_feature, **kwargs)


        loss_dict = dict()
        x, depth_real, depth_vitual = self.img_view_transformer(
            [img_feature, rots, trans, intrins, post_rots, post_trans, bda, mlp_input, intri_actually])


        bev_feats_source = self.bev_encoder(x)
        # source 2d data
        loss_heatmaps_source, loss_regess_source, heatmaps_source = self.img_aux(img_feature, **kwargs)
        # source bev data
        BEV_features_source, unused_loss_bev_source = self.img_view_transformer.get_BEV_feats_from_voxel(bev_feats_source)
        PV_features, unused_loss = self.img_view_transformer.get_PV_feats(bev_feats_source)
        PV_features_aug, unused_loss_aug = self.img_view_transformer.get_PV_feats_aug(bev_feats_source, [intrins, post_rots, post_trans, bda, intri_actually], **kwargs)
        loss_pv_heatmaps, loss_pv_regess, pv_heatmaps = self.bev_img_aux(PV_features, mlp_input, 'img', depth_flag='real', **kwargs)
        loss_pv_heatmaps_aug, loss_pv_regess_aug, pv_heatmaps_aug = self.bev_img_aux(PV_features_aug, mlp_input_aug,'img_aug', depth_flag='real', **kwargs)
        bev_heatmaps, bev_unused_loss = self.bev_img_aux.get_heatmaps(BEV_features_source)
        # consistency
        loss_dict.update({'loss_heatmaps_source': loss_heatmaps_source})
        loss_dict.update({'loss_regess_source': loss_regess_source})
        loss_dict.update({'loss_pv_heatmaps_source': loss_pv_heatmaps})
        loss_dict.update({'loss_pv_regess_source': loss_pv_regess})
        loss_dict.update({'loss_pv_heatmaps_aug_source': loss_pv_heatmaps_aug})
        loss_dict.update({'loss_pv_regess_aug_source': loss_pv_regess_aug})
        loss_dict.update({'loss_large_2D': cosim_loss})
        loss_dict.update({'loss_biased': loss_biased_0+loss_biased_1})

        loss_dict.update({'loss_unused_xxx': 0.0*(unused_loss_bev_source+unused_loss+unused_loss_aug+bev_unused_loss
                                                )})
        pts_feats = None
        return [bev_feats_source], bev_heatmaps,  pts_feats, depth_real, depth_vitual, loss_dict


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _, _= self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['img_metas'] = img_metas
        return bbox_list  # img_metas #

    def loss_reweigt(self, losses):

        if self.loss_weight == None:
            return losses
        else:
            losses['loss_depth_source'] = self.loss_weight[0] * losses['loss_depth_source']
            losses['loss_large_2D'] = self.loss_weight[1] * losses['loss_large_2D']
            losses['loss_heatmaps_source'] = self.loss_weight[2] * losses['loss_heatmaps_source']
            losses['loss_regess_source'] = self.loss_weight[3] * losses['loss_regess_source']


            losses['loss_pv_heatmaps_source'] = self.loss_weight[4] * losses['loss_pv_heatmaps_source']
            losses['loss_pv_regess_source'] = self.loss_weight[5] * losses['loss_pv_regess_source']
            losses['loss_pv_heatmaps_aug_source'] = self.loss_weight[6] * losses['loss_pv_heatmaps_aug_source']
            losses['loss_pv_regess_aug_source'] = self.loss_weight[7] * losses['loss_pv_regess_aug_source']
            losses['loss_bev_heatmap_source'] = self.loss_weight[8] * losses['loss_bev_heatmap_source']
            losses['loss_biased'] = self.loss_weight[9] * losses['loss_biased']

        return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):


        img_feats, bev_heatmaps, pts_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(
            points, img=img_inputs, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)

        gt_depth = kwargs['ann_maps_2d']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth_vitual)
        losses = dict(loss_depth_source=loss_depth)
        losses.update(loss_dict)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas, bev_heatmap=bev_heatmaps,
                                            pseudo_flag=False, gt_bboxes_ignore=None)
        losses.update(losses_pts)


        return self.loss_reweigt(losses)