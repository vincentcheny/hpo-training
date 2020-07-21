#!/bin/bash
#SBATCH --job-name=DF10h
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/zhuoming.wu/inception10GB/dragonfly.log
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --nodelist=chpc-k80gpu001
#SBATCH --cpus-per-task=4

python train-dragonfly.py