cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

bash ./tools/dist_train.sh  ./configs/Single_BEVDet_v2/bevdet-r50-da.py 8
