cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV






# 单数据集lyft训练，在三个测试
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-nus.py ./work_dirs/pdbev-r50-waymo/epoch_14.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py ./work_dirs/pdbev-r50-waymo/epoch_14.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-lyft.py ./work_dirs/pdbev-r50-waymo/epoch_14.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

# 单数据集waymo训练，在三个测试
#CUDA_VISIBLE_DEVICES=4,5,6,7
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-nus.py ./work_dirs/pdbev-r50-waymo/epoch_20.pth  4 --eval bbox --out ./work_dirs/pdbev-r50-waymo/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py ./work_dirs/pdbev-r50-waymo/epoch_20.pth  4 --eval bbox --out ./work_dirs/pdbev-r50-waymo/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-lyft.py ./work_dirs/pdbev-r50-waymo/epoch_20.pth  4 --eval bbox --out ./work_dirs/pdbev-r50-waymo/DA2lyft.pkl

bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-nus.py ./work_dirs/pdbev-r50-cbgs-uni/epoch_10.pth  8 --eval bbox --out ./work_dirs/pdbev-r50-waymo/DA2lyft.pkl
bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-waymo.py ./work_dirs/pdbev-r50-cbgs-uni/epoch_10.pth  8 --eval bbox --out ./work_dirs/pdbev-r50-waymo/DA2lyft.pkl
bash ./tools/dist_test.sh  ./configs/Single_BEVDet/bevdet-r50-lyft.py ./work_dirs/pdbev-r50-cbgs-uni/epoch_10.pth  8 --eval bbox --out ./work_dirs/pdbev-r50-waymo/DA2lyft.pkl
