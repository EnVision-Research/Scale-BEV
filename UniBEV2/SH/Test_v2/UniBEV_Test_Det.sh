cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV




# 全部数据集v2训练v2测试，在三个测试
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  ./work_dirs/bevdet-r50-uni_v2/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py ./work_dirs/bevdet-r50-uni_v2/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py ./work_dirs/bevdet-r50-uni_v2/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl


#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  ./work_dirs/pdbev-r50-uni_v2/epoch_17.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py ./work_dirs/pdbev-r50-uni_v2/epoch_17.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py ./work_dirs/pdbev-r50-uni_v2/epoch_17.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl


#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  ./work_dirs/pdbev-r50-uni_v2/epoch_17.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py ./work_dirs/pdbev-r50-uni_v2/epoch_17.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py ./work_dirs/pdbev-r50-uni_v2/epoch_17.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  ./work_dirs/bevdet-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py ./work_dirs/bevdet-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py ./work_dirs/bevdet-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

##
##
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py ./work_dirs/pdbev-sam-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py ./work_dirs/pdbev-sam-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py ./work_dirs/pdbev-sam-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl




bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
./work_dirs/bevdet-r50-uni_v2/epoch_20.pth  8  \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_det_nui2nus20.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
./work_dirs/bevdet-r50-uni_v2/epoch_20.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_det_nui2waymo20.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
./work_dirs/bevdet-r50-uni_v2/epoch_20.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_det_nui2lyft20.txt


#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/bevdepth-r50-uni_v2/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_bevdepth_nui2nus20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/bevdepth-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_bevdepth_nui2waymo20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/bevdepth-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_bevdepth_nui2lyft20.txt

#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/bevdet-r50-uni_v2/epoch_18.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_det_nui2nus18.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/bevdet-r50-uni_v2/epoch_18.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_det_nui2waymo18.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/bevdet-r50-uni_v2/epoch_18.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_det_nui2lyft18.txt
#
#
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/bevdet-r50-uni_v2/epoch_16.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_det_nui2nus16.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/bevdet-r50-uni_v2/epoch_16.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_det_nui2waymo16.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/bevdet-r50-uni_v2/epoch_16.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_det_nui2lyft16.txt
#
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/bevdet-r50-uni_v2/epoch_10.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_det_nui2nus10.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/bevdet-r50-uni_v2/epoch_10.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_det_nui2waymo10.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/bevdet-r50-uni_v2/epoch_10.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_det_nui2lyft10.txt