cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV
bash ./tools/dist_train.sh  ./configs/Uni_BEV/bevdet-r50-cbgs-uni.py 8 --checkpoint ./work_dirs/bevdet-r50-cbgs-uni/epoch_14.pth