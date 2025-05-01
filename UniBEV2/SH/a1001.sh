source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate bevformer_a100
cd /mnt/cfs/algorithm/hao.lu/mmdetection3d_a100
pip install -v -e .
cd /mnt/cfs/algorithm/hao.lu/Code/UniBEV
bash ./tools/dist_train.sh  ./configs/Uni_BEV_v2/bevdet-r50-uni_v2.py  4