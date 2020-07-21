#!/bin/bash
#SBATCH --job-name=vgg16_tpe
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=wuzhm7@gmail.com
#SBATCH --output=/lustre/project/EricLo/zhuoming.wu/vgg16-cifar10-tf2/tpe.log
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --nodelist=chpc-gpu002
#SBATCH --cpus-per-task=2

nnictl create --config config.yml -p 8087
sleep 20h
