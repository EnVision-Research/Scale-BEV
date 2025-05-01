cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB

cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV



#bash ./tools/dist_test.sh ./configs/Single_BEVDepth_v2/bevdepth-r50-nus-onlycar.py  \
#./work_dirs/bevdepth-r50-waymo-onlycar/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/Onlycar_BEVDepth_way2nus.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDepth_v2/bevdepth-r50-waymo-onlycar.py  \
#./work_dirs/bevdepth-r50-nus-onlycar/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/Onlycar_BEVDepth_nus2waymo.txt

bash ./tools/dist_test.sh ./configs/Single_BEVDepth_v2/bevdepth-r50-nus-onlycar.py  \
./work_dirs/bevdepth-r50-waymo-onlycar/epoch_6.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/Onlycar_BEVDepth_way2nus6.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDepth_v2/bevdepth-r50-waymo-onlycar.py  \
./work_dirs/bevdepth-r50-nus-onlycar/epoch_6.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/Onlycar_BEVDepth_nus2waymo6.txt


bash ./tools/dist_test.sh ./configs/Single_BEVDepth_v2/bevdepth-r50-nus-onlycar.py  \
./work_dirs/bevdepth-r50-waymo-onlycar/epoch_12.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/Onlycar_BEVDepth_way2nus12.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDepth_v2/bevdepth-r50-waymo-onlycar.py  \
./work_dirs/bevdepth-r50-nus-onlycar/epoch_12.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/Onlycar_BEVDepth_nus2waymo12.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py  \
#./work_dirs/bevdet-r50-nus/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_BEVDet_nus2lyft.txt
#
## single waymo2X
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  \
#./work_dirs/bevdet-r50-waymo/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_BEVDet_waymo2nus.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo.py  \
#./work_dirs/bevdet-r50-waymo/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_BEVDet_waymo2waymo.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py  \
#./work_dirs/bevdet-r50-waymo/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_BEVDet_waymo2lyft.txt

# single lyft2X

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py  \
./work_dirs/bevdet-r50-lyft/epoch_20.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/single_BEVDet_lyft2lyft.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  \
./work_dirs/bevdet-r50-lyft/epoch_20.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/single_BEVDet_lyft2nus.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo.py  \
./work_dirs/bevdet-r50-lyft/epoch_20.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/single_BEVDet_lyft2waymo.txt


