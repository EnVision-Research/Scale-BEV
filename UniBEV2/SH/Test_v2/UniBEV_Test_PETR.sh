cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV


#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-nus.py ./work_dirs/petr-r50-lyft/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-waymo.py ./work_dirs/petr-r50-lyft/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-lyft.py ./work_dirs/petr-r50-lyft/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
##
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-nus.py ./work_dirs/petr-r50-cbgs-single-nus/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-waymo.py ./work_dirs/petr-r50-cbgs-single-nus/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-lyft.py ./work_dirs/petr-r50-cbgs-single-nus/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-nus.py ./work_dirs/petr-r50-waymo/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-waymo.py ./work_dirs/petr-r50-waymo/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_PETR/petr-r50-lyft.py ./work_dirs/petr-r50-waymo/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

# bash ./tools/dist_test.sh  ./configs/Single_PETR_V2/petr-r50-lyft.py \
# ./work_dirs/petr-r50-uni_v2/epoch_20.pth  8 --eval bbox \
# --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
# > ./log/uni-petr-2lyft.txt


# bash ./tools/dist_test.sh  ./configs/Single_PETR_V2/petr-r50-nus.py \
# ./work_dirs/petr-r50-uni_v2/epoch_20.pth  8 --eval bbox \
# --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
# > ./log/uni-petr-2nus.txt

# bash ./tools/dist_test.sh  ./configs/Single_PETR_V2/petr-r50-waymo.py \
# ./work_dirs/petr-r50-uni_v2/epoch_20.pth  8 --eval bbox \
# --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
# > ./log/uni-petr-2way.txt


bash ./tools/dist_test.sh  ./configs/Single_PETR_V2/petr-r50-nus.py \
./work_dirs/petr-r50-uni_v2/epoch_20.pth 8 --format-only --eval-options jsonfile_prefix="./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/"


# bash ./tools/dist_test.sh  ./configs/Single_PETR_V2/petr-r50-nus.py \
# ./work_dirs/petr-r50-uni_v2/epoch_20.pth  8 --eval bbox \
# --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2nus.pkl \
# > ./log/uni-petr-2nus.txt

# bash ./tools/dist_test.sh  ./configs/Single_PETR_V2/petr-r50-waymo.py \
# ./work_dirs/petr-r50-uni_v2/epoch_20.pth  8 --eval bbox \
# --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2waymo.pkl \
# > ./log/uni-petr-2way.txt





# Uni_BEV 测试不对
# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Uni_test/bevdet-r50-cbgs-uni.py ./work_dirs/bevdet-r50-cbgs-uni/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
