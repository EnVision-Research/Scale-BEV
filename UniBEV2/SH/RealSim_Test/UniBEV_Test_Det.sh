cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV




### da2da
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-da-test.py \
#./work_dirs/bevdet-r50-da/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/realsim_det_da2da20.txt


## bevdet uni2
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-da-test.py  \
#./work_dirs/bevdet-r50-realsim-v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/realsim_det_uni2da20.txt

#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/bevdet-r50-realsim-v2/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/realsim_det_nui2nus20.txt

# bevdepth uni2

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-da-test.py  \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/bevdepth-r50-realsim-v2/epoch_20.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/realsim_detpth_uni2da20.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/bevdepth-r50-realsim-v2/epoch_20.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/realsim_detpth_uni2nus20.txt


## ourt uni2
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-da-test.py  \
#./work_dirs/pdbev-r50-realsim-v2/epoch_20.pth  8 \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/realsim_pdbev_uni2da20.txt


#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py \
#./work_dirs/pdbev-r50-realsim-v2/epoch_20.pth  8  \
#--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/realsim_pdbev_nui2nus20.txt




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