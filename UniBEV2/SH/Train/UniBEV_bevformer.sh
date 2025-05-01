cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

bash ./tools/dist_train.sh  ./configs/Uni_BEV/bevformer-r50-uni.py 8

