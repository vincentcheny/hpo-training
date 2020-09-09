#!/bin/bash
#SBATCH --job-name=resnet_real
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/hpo-training/replaceable-tf2_resnet_mobilenet/dragonfly_resnet.log
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --nodelist=chpc-gpu002
#SBATCH --cpus-per-task=2

python dragonfly_train.py
