cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate FB
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV

#bash ./tools/dist_train.sh  ./configs/Uni_BEV/petr-r50-cbgs-uni.py 8
#bash ./tools/dist_train.sh  ./configs/Uni_BEV/bevformer-r50-cbgs-uni.py 8
bash ./tools/dist_train.sh  ./configs/Uni_BEV_v2/pdbev-sam-r50-uni_v2.py 8
#bash ./tools/dist_train.sh  ./configs/Uni_BEV/bevdet-r50-cbgs-uni.py 8

#bash ./tools/dist_train.sh   ./configs/Uni_BEV/bevdepth-r50-cbgs-uni-nus.py 8

# bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2NUS-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-base/epoch_17.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

# bash ./tools/dist_test.sh  /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2NUS-dg-256.py  /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-base/epoch_17.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

# bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-base-256/epoch_24.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-base/epoch_12.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-base/epoch_14.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-base/epoch_16.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-base/epoch_18.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-base/epoch_20.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl


#bash ./tools/dist_test.sh  ./configs/1_DA_Debug_refine_test/bevdet-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdet-r50-cbgs-NUS2X-dg-base/epoch_24.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/1_DA_Debug_refine_test/bevdet-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdet-r50-cbgs-NUS2X-dg-base/epoch_24.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/1_DA_Debug_refine_test/bevdet-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdet-r50-cbgs-NUS2X-dg-base/epoch_24.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh  ./configs/1_DA_Debug_refine_test/bevdet-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdet-r50-cbgs-NUS2X-dg-base/epoch_24.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-2D/epoch_1.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-2D/epoch_2.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-2D/epoch_3.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-2D/epoch_4.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-NUS2X-dg-2D/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf/epoch_4.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf/epoch_3.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf/epoch_2.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf/epoch_1.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl



#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf1/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf1/epoch_4.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf1/epoch_3.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf1/epoch_2.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf1/epoch_1.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl



#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf2/epoch_8.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf2/epoch_6.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf2/epoch_4.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf2/epoch_2.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf2/epoch_1.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf3/epoch_7.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf3/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf3/epoch_3.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf3/epoch_2.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf3/epoch_1.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#


#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdet-r50-cbgs-NUS2LYFT-Base.py ./work_dirs/bevdet-r50-cbgs-NUS2X-dg-base/epoch_24.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdet-r50-cbgs-NUS2LYFT-Base.py ./work_dirs/bevdet-r50-cbgs-NUS2X-dg-base/epoch_18.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdet-r50-cbgs-NUS2LYFT-Base.py ./work_dirs/bevdet-r50-cbgs-NUS2X-dg-base/epoch_16.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdet-r50-cbgs-NUS2LYFT-Base.py ./work_dirs/bevdet-r50-cbgs-NUS2X-dg-base/epoch_14.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdet-r50-cbgs-NUS2LYFT-Base.py ./work_dirs/bevdet-r50-cbgs-NUS2X-dg-base/epoch_12.pth  8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl


#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_1.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_7.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_3.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_2.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_1.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_7.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_3.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf5/epoch_2.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf6/epoch_1.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf6/epoch_7.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf6/epoch_5.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf6/epoch_3.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdepth-r50-cbgs-NUS2X-dg-nerf6/epoch_2.pth  4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-uda-NUS2LYFT/epoch_8.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-uda-NUS2LYFT/epoch_4.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-uda-NUS2LYFT/epoch_2.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-uda-NUS2LYFT/epoch_1.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl


#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-dg/epoch_5.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-dg/epoch_4.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-dg/epoch_3.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-dg/epoch_2.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl


#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/epoch_10.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/epoch_9.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/epoch_8.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/epoch_7.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/epoch_6.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevdepth-r50-cbgs-DA2LYFT-Base.py /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_verse/work_dirs/bevdepth-r50-cbgs/epoch_24.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_1.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_1.pth 8 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_2.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_2.pth 8 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_3.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_3.pth 8 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_4.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_4.pth 8 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_5.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_5.pth 8 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_6.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_6.pth 8 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_7.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-r50-cbgs-NUS2X-2d/epoch_7.pth 8 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl


#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d-ab3/epoch_11.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d-ab3/epoch_10.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d-ab3/epoch_8.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py  ./work_dirs/bevdy-r50-cbgs-NUS2X-2d-ab3/epoch_6.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl


#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdy-uda-NUS2LYFT/epoch_3.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-uda-NUS2LYFT/epoch_3.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdy-uda-NUS2LYFT-ab0/epoch_3.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-uda-NUS2LYFT-ab0/epoch_3.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdy-uda-NUS2LYFT/epoch_1.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-uda-NUS2LYFT/epoch_1.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevdy-uda-NUS2LYFT-ab0/epoch_1.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevdy-uda-NUS2LYFT-ab0/epoch_1.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#


#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_23.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#
#bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_23.pth 8 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
#bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_22.pth 8 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#
#bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_22.pth 8 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#
##
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_20.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_20.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_19.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_19.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_17.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_17.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_15.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_15.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_13.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final-wpre/epoch_13.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl

#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2NUS-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final/epoch_23.pth 4 --eval bbox --out ./work_dirs/bevbox-r50-cbgs-NUS2X-2d/NUS2NUS.pkl
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh ./configs/1_DA_Debug_refine_test/bevbox-r50-cbgs-NUS2LYFT-dg.py ./work_dirs/bevbox-r50-cbgs-NUS2X-2d-final/epoch_23.pth 4 --eval bbox --out ./work_dirs/bevdet-r50-cbgs-NUS2X-intri-nobn/DA2lyft.pkl
#

# bevbox-r50-cbgs-NUS2X-dg
