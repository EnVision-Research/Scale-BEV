cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

bash ./tools/dist_train.sh  ./configs/Single_BEVDepth/bevdepth-r50-lyft.py 8
bash ./tools/dist_train.sh  ./configs/Single_BEVDepth/bevdepth-r50-nus.py 8
bash ./tools/dist_train.sh  ./configs/Single_BEVDepth/bevdepth-r50-waymo.py 8