cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV


#bash ./tools/dist_train.sh  ./configs/Uni_BEV_v2/pdbev-samv2-aug-p13-r50-uni_v2.py 8
bash ./tools/dist_train.sh  /mnt/cfs/algorithm/hao.lu/Code/UniBEV/configs/Uni_BEV_v2/pdbev-samv2-aug-p9-r50-uni_v2.py 4
#bash ./tools/dist_train.sh  ./configs/Uni_BEV_v2/pdbev-r50-uni_v2.py 8  #--checkpoint ./work_dirs/pdbev-r50-cbgs-uni/epoch_20.pth


# bash ./tools/dist_train.sh  ./configs/Uni_BEV_v2/pdbev-samv2-r50-uni_v2.py 8