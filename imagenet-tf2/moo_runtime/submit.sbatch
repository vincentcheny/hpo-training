#!/bin/bash
#SBATCH --job-name=imagenet_runtime
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/research/dept7/cyliu/moo/chen.yu/log/runtime.log
#SBATCH --gres=gpu:2
#SBATCH -w gpu17
#SBATCH -p batch_72h

unset http_proxy
unset https_proxy
nnictl create --config config_bohb.yml -p 8062 --debug
sleep 3d
