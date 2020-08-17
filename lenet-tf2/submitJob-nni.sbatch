#!/bin/bash
#SBATCH --job-name=lenet_soo
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/research/dept7/cyliu/moo/chen.yu/hpo-training/lenet-tf2/soo.log
#SBATCH --gres=gpu:3
#SBATCH -w gpu37
#SBATCH -p batch_72h

unset http_proxy
unset https_proxy
nnictl create --config config-bohb-soo.yml -p 8062 --debug
sleep 1h
