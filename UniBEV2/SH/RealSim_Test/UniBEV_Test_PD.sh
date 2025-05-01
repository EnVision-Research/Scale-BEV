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

#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py ./work_dirs/pdbev-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py ./work_dirs/pdbev-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py ./work_dirs/pdbev-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#


bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
./work_dirs/pdbev-samv2-r50-uni_v2/epoch_20.pth  8  \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_pd_sam2_nui2nus20.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
./work_dirs/pdbev-samv2-r50-uni_v2/epoch_20.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_pd_sam2_nui2waymo20.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
./work_dirs/pdbev-samv2-r50-uni_v2/epoch_20.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_pd_sam2_nui2lyft20.txt



#/mnt/cfs/algorithm/hao.lu/Code/UniBEV/work_dirs/pdbev-samv2-r50-uni_v2/epoch_17.pth
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_18.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2nus18.txt

#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2waymo18.txt

#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_18.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2lyft18.txt


#
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_16.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2nus16.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_16.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2waymo16.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_16.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2lyft16.txt
#
#
#
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_10.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2nus10.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_10.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2waymo10.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_10.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2lyft10.txt



#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/pdbev-r50-uni_v2/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_nui2nus.txt

#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/pdbev-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_nui2waymo.txt

#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/pdbev-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_nui2lyft.txt


