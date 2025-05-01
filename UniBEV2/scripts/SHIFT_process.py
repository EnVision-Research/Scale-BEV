import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import mmcv

SHIFT_source = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images'
SHIFT_target = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT_Pro'


