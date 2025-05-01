cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV


# ab测试
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/pdbev-samv2-aug-p6-r50-uni_v2/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab6_nui2nus20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/pdbev-samv2-aug-p6-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab6_nui2waymo20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/pdbev-samv2-aug-p6-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab6_nui2lyft20.txt

# ab1
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab8_nui2nus20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab8_nui2waymo20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab8_nui2lyft20.txt

## ab2
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_13.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab8_nui2nus20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_13.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab8_nui2waymo20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_13.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab8_nui2lyft20.txt

# ab3
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p9-r50-uni_v2/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab9_nui2nus20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p9-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab9_nui2waymo20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p9-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab9_nui2lyft20.txt
#
## ab4
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/pdbev-samv2-aug-p4-r50-uni_v2/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab4_nui2nus20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/pdbev-samv2-aug-p4-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab4_nui2waymo20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/pdbev-samv2-aug-p4-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_ab4_nui2lyft20.txt

#
## samv2测试
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2nus20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2waymo20.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
#./work_dirs/pdbev-samv2-r50-uni_v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/uni_pd_sam2_nui2lyft20.txt



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


bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
./work_dirs/pdbev-r50-uni_v2/epoch_20.pth 8 --format-only --eval-options jsonfile_prefix="./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/pd_bev"


# bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
# ./work_dirs/pdbev-r50-uni_v2/epoch_20.pth  8  \
# --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft_pd.pkl \
# > ./log/uni_pd_nui2nus.txt

# bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo-5.py  \
# ./work_dirs/pdbev-r50-uni_v2/epoch_20.pth  8 \
# --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
# > ./log/uni_pd_nui2waymo.txt

# bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
# ./work_dirs/pdbev-r50-uni_v2/epoch_20.pth  8 \
# --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
# > ./log/uni_pd_nui2lyft.txt


