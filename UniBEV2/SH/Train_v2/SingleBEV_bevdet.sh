cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

#CUDA_VISIBLE_DEVICES=4,5,6,7  bash ./tools/dist_train.sh  ./configs/Single_BEVDet/bevdet-r50-cbgs-lyft.py 4

#bash ./tools/dist_train.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py 8
#bash ./tools/dist_train.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo.py 8
#bash ./tools/dist_train.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-onlycar.py 8
#bash ./tools/dist_train.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus-onlycar.py 8
#bash ./tools/dist_train.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py 8



bash ./tools/dist_train.sh  /mnt/cfs/algorithm/hao.lu/Code/UniBEV/configs/Single_PDBEV_v2/pdbev-r50-nus-onlycar.py 8
bash ./tools/dist_train.sh  /mnt/cfs/algorithm/hao.lu/Code/UniBEV/configs/Single_PDBEV_v2/pdbev-r50-waymo-onlycar.py 8