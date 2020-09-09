#!/bin/bash
#SBATCH --job-name=bt_df
#SBATCH --mail-type=END, FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/output_bt_df.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

python train-dragonfly.py