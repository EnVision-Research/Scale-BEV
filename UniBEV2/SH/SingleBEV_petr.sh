cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

bash ./tools/dist_train.sh  ./configs/Single_PETR/petr-r50-cbgs-single-lyft.py  8
bash ./tools/dist_train.sh  ./configs/Single_PETR/petr-r50-cbgs-single-nus.py   8
bash ./tools/dist_train.sh  ./configs/Single_PETR/petr-r50-cbgs-single-waymo.py 8