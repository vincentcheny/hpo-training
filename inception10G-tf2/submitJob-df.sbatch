#!/bin/bash
#SBATCH --job-name=df_inception
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/research/dept7/ychen/model/hpo-training/inception10G-tf2/slurm-log/0-Dragonfly.log
#SBATCH --gres=gpu:2
#SBATCH -w gpu21
#SBATCH -p batch_72h

python train-dragonfly.py  --seed 3000 --log_path ./0-Dragonfly.log
