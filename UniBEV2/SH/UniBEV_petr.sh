cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

bash ./tools/dist_train.sh  ./configs/Uni_BEV/petr-r50-cbgs-uni.py 8 #--checkpoint /mnt/cfs/algorithm/hao.lu/Code/UniBEV/work_dirs/petr-r50-cbgs-uni/epoch_5.pth
