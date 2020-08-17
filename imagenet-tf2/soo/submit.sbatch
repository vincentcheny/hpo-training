#!/bin/bash
#SBATCH --job-name=imagenet_soo
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/research/dept7/cyliu/moo/chen.yu/log/soo_gpu37.log
#SBATCH --gres=gpu:2
#SBATCH -w gpu37
#SBATCH -p batch_72h

unset http_proxy
unset https_proxy
nnictl create --config config_bohb.yml -p 8127 --debug
sleep 3d
