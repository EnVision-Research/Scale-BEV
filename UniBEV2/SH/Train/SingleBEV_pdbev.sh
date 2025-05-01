cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

#CUDA_VISIBLE_DEVICES=4,5,6,7  bash ./tools/dist_train.sh  ./configs/Single_BEVDet/bevdet-r50-cbgs-lyft.py 4

bash ./tools/dist_train.sh  ./configs/Single_PDBEV/pdbev-r50-lyft.py 8
bash ./tools/dist_train.sh  ./configs/Single_PDBEV/pdbev-r50-waymo.py 8
bash ./tools/dist_train.sh  ./configs/Single_PDBEV/pdbev-r50-nus.py 8
