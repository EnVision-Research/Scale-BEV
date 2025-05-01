cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc1

# 复制的环境
conda activate FB_A1001

# 这个是复制的mmdet3d
cd /mnt/cfs/algorithm/hao.lu/mmdetection3d_a10011
python setup.py develop

# 运行命令
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV
bash ./tools/dist_train.sh  ./configs/Uni_BEV_v2/bevdet-r50-uni_v2.py  4

