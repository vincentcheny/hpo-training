#!/bin/bash
#SBATCH --job-name=BOHB10h
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/zhuoming.wu/inception10GB/bohb.log
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --nodelist=chpc-k80gpu001
#SBATCH --cpus-per-task=4

nnictl create --config config-bohb.yml -p 8101
sleep 20h