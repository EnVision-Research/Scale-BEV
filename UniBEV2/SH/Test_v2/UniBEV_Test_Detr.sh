cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB



cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV


bash ./tools/dist_test.sh  ./configs/Single_DETR3D_v2/detr-r50-nus.py \
./work_dirs/detr-r50-uni_v2/epoch_20.pth  8  \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_detr_nui2nus20.txt

bash ./tools/dist_test.sh  ./configs/Single_DETR3D_v2/detr-r50-waymo.py  \
./work_dirs/detr-r50-uni_v2/epoch_20.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_detr_nui2waymo20.txt

bash ./tools/dist_test.sh  ./configs/Single_DETR3D_v2/detr-r50-lyft.py \
./work_dirs/detr-r50-uni_v2/epoch_20.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_detr_nui2lyft20.txt
