cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV


#bash ./tools/dist_train.sh  ./configs/Uni_BEV_v2/petr-r50-nus-lyft_v2.py  8 --checkpoint  ./work_dirs/petr-r50-cbgs-single-nus/epoch_20.pth
#bash ./tools/dist_train.sh  ./configs/Uni_BEV/petr-r50-cbgs-uni.py 8 # --checkpoint  ./work_dirs/petr-r50-cbgs-single-nus/epoch_20.pth

bash ./tools/dist_train.sh  ./configs/Uni_BEV_v2/petr-r50-uni_v2.py 8