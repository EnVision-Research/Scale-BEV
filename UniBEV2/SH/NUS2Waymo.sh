cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/bevdet_our/bevdepth-r50-cbgs-pc-waymo.py ./work_dirs/bevdepth-r50-cbgs-pc-waymo/epoch_23.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/bevdet_our/bevdepth-r50-cbgs-pc-lyft.py ./work_dirs/bevdepth-r50-cbgs-pc-waymo/epoch_23.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/bevdet_our/bevdepth-r50-cbgs-pc-nus.py ./work_dirs/bevdepth-r50-cbgs-pc-waymo/epoch_23.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#bash ./tools/dist_test.sh  ./configs/bevdet_our/bevdepth-r50-cbgs-pc-lyft.py ./work_dirs/bevdepth-r50-cbgs-pc-waymo/epoch_23.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#bash ./tools/dist_test.sh  ./configs/bevdet_our/bevdepth-r50-cbgs-pc-nus.py ./work_dirs/bevdepth-r50-cbgs-pc-waymo/epoch_23.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/bevdet_our/bevdepth-r50-cbgs-pc-lyft.py /mnt/cfs/algorithm/hao.lu/Code/UniBEV/Models/NUS2LYFT.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#bash ./tools/dist_test.sh  ./configs/bevdet_our/bevdepth-r50-cbgs-pc-waymo.py /mnt/cfs/algorithm/hao.lu/Code/UniBEV/Models/NUS2LYFT.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_train.sh   /mnt/cfs/algorithm/hao.lu/Code/UniBEV/configs/Uni_BEV/bevdepth-r50-cbgs-uni-nus.py

