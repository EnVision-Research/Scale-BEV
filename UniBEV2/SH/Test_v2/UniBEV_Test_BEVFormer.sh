cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV



#bash ./tools/dist_test.sh  ./configs/Single_FB_BEVFormer_V2/fb-bevformer-r50-nus.py  \
#./work_dirs/fb-bevformer-r50-uni_v2/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/bevformer-r50-uni2nux.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_FB_BEVFormer_V2/fb-bevformer-r50-waymo.py  \
#./work_dirs/fb-bevformer-r50-uni_v2/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/bevformer-r50-uni2waymo.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_FB_BEVFormer_V2/fb-bevformer-r50-lyft.py \
#./work_dirs/fb-bevformer-r50-uni_v2/epoch_20.pth  8 --eval bbox \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
#> ./log/bevformer-r50-uni2lyft.txt


bash ./tools/dist_test.sh  ./configs/Single_BEVFormer_V2/bevformer-r50-nus.py  \
./work_dirs/bevformer-r50-uni_v2/epoch_20.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/bevformer-1-r50-uni2nux.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVFormer_V2/bevformer-r50-waymo.py  \
./work_dirs/bevformer-r50-uni_v2/epoch_20.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/bevformer-1-r50-uni2waymo.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVFormer_V2/bevformer-r50-lyft.py \
./work_dirs/bevformer-r50-uni_v2/epoch_20.pth  8 --eval bbox \
--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/bevformer-1-r50-uni2lyft.txt

#bash ./tools/dist_test.sh  ./configs/Single_FB_V2/fb-r50-nus.py   \
#./work_dirs/fb-r50-uni_v2/epoch_20.pth  8 --eval bbox  \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl   \
#> ./log/fb-r50-uni2nux.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_FB_V2/fb-r50-waymo.py   \
#./work_dirs/fb-r50-uni_v2/epoch_20.pth  8 --eval bbox   \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl   \
#> ./log/fb-r50-uni2waymo.txt
#
#bash ./tools/dist_test.sh  ./configs/Single_FB_V2/fb-r50-lyft.py  \
#./work_dirs/fb-r50-uni_v2/epoch_20.pth  8 --eval bbox   \
#--out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl  \
#> ./log/fb-r50-uni2lyft.txt

#
#bash ./tools/dist_test.sh  ./configs/Single_BEVFormer_V2/bevformer-r50-waymo.py ./work_dirs/bevformer-r50-uni_v2/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/Single_BEVFormer_V2/bevformer-r50-lyft.py ./work_dirs/bevformer-r50-uni_v2/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#
#bash ./tools/dist_test.sh  ./configs/Single_FB_BEVFormer_V2/fb-bevformer-r50-nus.py  ./work_dirs/fb-bevformer-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/Single_FB_BEVFormer_V2/fb-bevformer-r50-waymo.py ./work_dirs/fb-bevformer-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/Single_FB_BEVFormer_V2/fb-bevformer-r50-lyft.py ./work_dirs/fb-bevformer-r50-uni_v2/epoch_20.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#
