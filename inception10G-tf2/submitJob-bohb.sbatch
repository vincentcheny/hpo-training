#!/bin/bash
#SBATCH --job-name=bohb_inception
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/research/dept7/ychen/model/inception10G/comparison/slurm-log/1-BOHB.log
#SBATCH --gres=gpu:2
#SBATCH -w gpu21
#SBATCH -p batch_72h

unset http_proxy
unset https_proxy
nnictl create --config config-bohb.yml -p 8130 --debug
sleep 3d
