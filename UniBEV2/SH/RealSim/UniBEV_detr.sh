cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

bash ./tools/dist_train.sh  ./configs/Uni_RealSim_v2/detr-r50-realsim_v2.py 8

