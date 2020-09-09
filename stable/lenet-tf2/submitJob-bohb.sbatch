#!/bin/bash
#SBATCH --job-name=bohb_lenet
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/uac/rshr/cyliu/bigDataStorage/moo/chen.yu/hpo-training/lenet-tf2/slurm-log/1-BOHB.log
#SBATCH --gres=gpu:2
#SBATCH -w gpu20
#SBATCH -p batch_72h

unset http_proxy
unset https_proxy
nnictl create --config config-bohb.yml -p 8132 --debug
sleep 3d
