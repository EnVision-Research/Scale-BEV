cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
#conda activate FB_A100


bash ./tools/dist_train.sh  ./configs/Uni_BEV_v2/bevdet-r50-uni_v2.py  8