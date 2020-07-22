#!/bin/bash
#SBATCH --job-name=lenet_df
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/hpo-training/lenet-tf2/dragonfly.log
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --nodelist=chpc-gpu002
#SBATCH --cpus-per-task=2

python train-dragonfly.py
