cd $(readlink -f `dirname $0`)
source /mnt/cfs/algorithm/yunpeng.zhang/.bashrc
conda activate bevdet_da
export PYTHONPATH="."

echo $1
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

bash tools/dist_train.sh $config $2 --no-validate ${@:3}