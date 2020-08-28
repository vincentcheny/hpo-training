#!/bin/bash
#SBATCH --job-name=df_vgg
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/research/dept7/ychen/model/hpo-training/vgg-tf2/slurm-log/0-Dragonfly.log
#SBATCH --gres=gpu:2
#SBATCH -w gpu19
#SBATCH -p batch_72h

python train-dragonfly.py --seed 1234 --log_path ./0-Dragonfly.log
