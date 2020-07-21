#!/bin/bash
#SBATCH --job-name=xception_df
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/hpo-training/replaceable2-tf2/dragonfly.log
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --nodelist=chpc-k80gpu002
#SBATCH --cpus-per-task=2

python dragonfly_train.py
