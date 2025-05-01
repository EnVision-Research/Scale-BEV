cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV




bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_1.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft1.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_2.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft2.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_3.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft3.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_4.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft4.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_5.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft5.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_6.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft6.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_7.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft7.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_8.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft8.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_9.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft9.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-p8-r50-uni_v2/epoch_10.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft10.pkl \
> ./log/uni_ab8_nui2lyft10.txt



bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_11.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft11.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_12.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft12.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_13.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft13.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_14.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft14.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_15.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft15.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_16.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft16.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_17.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft17.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_18.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft18.txt

bash ./tools/dist_test.sh  ./configs/Single_BEVDet_v2/bevdet-r50-lyft.py \
/mnt/cfs/algorithm/hao.lu/Code/UniBEV_A100/work_dirs/pdbev-samv2-aug-p8-r50-uni_v2/epoch_19.pth  8 \
--eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl \
> ./log/uni_ab8_nui2lyft19.txt
