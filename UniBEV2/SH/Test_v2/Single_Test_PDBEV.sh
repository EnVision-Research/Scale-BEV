cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV



# single nus2X
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  \
#./work_dirs/pdbev-r50-nus/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_pdbev_nus2nus.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo.py  \
./work_dirs/pdbev-r50-nus/epoch_20.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/single_pdbev_nus2waymo.txt

#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py  \
#./work_dirs/pdbev-r50-nus/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_pdbev_nus2lyft.txt
#
## single waymo2X
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  \
#./work_dirs/pdbev-r50-waymo/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_pdbev_waymo2nus.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo.py  \
#./work_dirs/pdbev-r50-waymo/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_pdbev_waymo2waymo.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py  \
#./work_dirs/pdbev-r50-waymo/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_pdbev_waymo2lyft.txt
#
## single lyft2X
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-nus.py  \
#./work_dirs/pdbev-r50-nus/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_pdbev_nus2nus.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-waymo.py  \
#./work_dirs/pdbev-r50-nus/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl\
#> ./log/single_pdbev_nus2waymo.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py  \
#./work_dirs/pdbev-r50-nus/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/single_pdbev_nus2lyft.txt
