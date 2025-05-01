# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DDetector
from .bevdet import BEVDepth4D, BEVDet, BEVDet4D, BEVDetTRT
from .centerpoint import CenterPoint
from .dynamic_voxelnet import DynamicVoxelNet
from .fcos_mono3d import FCOSMono3D
from .groupfree3dnet import GroupFree3DNet
from .h3dnet import H3DNet
from .imvotenet import ImVoteNet
from .imvoxelnet import ImVoxelNet
from .mink_single_stage import MinkSingleStage3DDetector
from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
from .mvx_two_stage import MVXTwoStageDetector
from .parta2 import PartA2
from .point_rcnn import PointRCNN
from .sassd import SASSD
from .single_stage_mono3d import SingleStageMono3DDetector
from .smoke_mono3d import SMOKEMono3D
from .ssd3dnet import SSD3DNet
from .votenet import VoteNet
from .voxelnet import VoxelNet
from .bevdepth import BEVDepth
from .pcbev import PCBEV_DG, PCBEV_UDA
from .obj_dgcnn import ObjDGCNN
from .detr3d import Detr3D
from .petr3d import Petr3D
from .petr3d_seg import Petr3D_seg
from .petr3d_depth import Petr3D_Depth, Petr3D_GTDepth
from .bevformer import BEVFormer
from .bevformer_fp16 import BEVFormer_fp16
from .bevformerV2 import BEVFormerV2
from .fbocc import FBOCC
from .scale_bev import Scale_BEV_Large2D


__all__ = [
    'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
    'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
    'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
    'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet', 'PointRCNN', 'SMOKEMono3D',
    'MinkSingleStage3DDetector', 'SASSD', 'BEVDet', 'BEVDet4D', 'BEVDepth4D',
    'BEVDetTRT', 'PCBEV_DG', 'PCBEV_UDA', 'BEVDepth', 'ObjDGCNN',
    'Detr3D', 'Petr3D', 'Petr3D_seg', 'Petr3D_Depth', 'Petr3D_GTDepth',
    'BEVFormer', 'BEVFormer_fp16', 'BEVFormerV2', 'FBOCC', 'Scale_BEV_Large2D'
]
