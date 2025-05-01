cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py  \
/mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed/work_dirs/pcbev-r50-cbgs-NUS2X-dg-18/epoch_23.pth  8  --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl  >  ./log/nus2waymo23.txt

#
bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py \
/mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed/work_dirs/pcbev-r50-cbgs-NUS2X-dg-18/epoch_6.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl  >  ./log/nus2waymo6.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py \
/mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed/work_dirs/pcbev-r50-cbgs-NUS2X-dg-18/epoch_10.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl  >  ./log/nus2waymo10.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py \
/mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed/work_dirs/pcbev-r50-cbgs-NUS2X-dg-18/epoch_16.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl  >  ./log/nus2waymo16.txt




#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-nus.py ./work_dirs/bevdet-r50-cbgs-uni/epoch_10.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py ./work_dirs/bevdet-r50-cbgs-uni/epoch_10.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-lyft.py ./work_dirs/bevdet-r50-cbgs-uni/epoch_10.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl


## 单数据集lyft训练，在三个测试
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-nus.py ./work_dirs/bevdet-r50-lyft/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py ./work_dirs/bevdet-r50-lyft/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-lyft.py ./work_dirs/bevdet-r50-lyft/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

## 单数据集nus训练，在三个测试
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-nus.py ./work_dirs/bevdet-r50-nus/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py ./work_dirs/bevdet-r50-nus/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-lyft.py ./work_dirs/bevdet-r50-nus/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#
## 单数据集waymo训练，在三个测试
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-nus.py ./work_dirs/bevdet-r50-waymo/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py ./work_dirs/bevdet-r50-waymo/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-lyft.py ./work_dirs/bevdet-r50-waymo/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl



# 全部数据集v2训练v2测试，在三个测试
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  ./work_dirs/bevdet-r50-uni_v2/epoch_10.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo.py ./work_dirs/bevdet-r50-uni_v2/epoch_10.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py ./work_dirs/bevdet-r50-uni_v2/epoch_10.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py ./work_dirs/bevdet-r50-uni_v2/epoch_10.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
