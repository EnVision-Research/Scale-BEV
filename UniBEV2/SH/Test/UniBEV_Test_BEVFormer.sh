cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV



CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-cbgs-single-lyft.py ./work_dirs/petr-r50-cbgs-uni/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-cbgs-single-waymo.py ./work_dirs/petr-r50-cbgs-uni/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-cbgs-single-lyft.py ./work_dirs/petr-r50-cbgs-uni/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
