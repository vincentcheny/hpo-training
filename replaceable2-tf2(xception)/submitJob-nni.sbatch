#!/bin/bash
#SBATCH --job-name=xception_tpe
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/hpo-training/replaceable2-tf2/tpe.log
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --nodelist=chpc-k80gpu002
#SBATCH --cpus-per-task=2

nnictl create --config config-tpe.yml -p 8090
sleep 20h
