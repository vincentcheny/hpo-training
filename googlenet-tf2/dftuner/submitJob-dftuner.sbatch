#!/bin/bash
#SBATCH --job-name=googlenet_dftuner
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/research/dept7/ychen/model/tf2_imagenet_noHierarchicalCopyAllReduce/dftuner/slurm-log/3-DFTuner1.log
#SBATCH --gres=gpu:2
#SBATCH -w gpu21
#SBATCH -p batch_72h

unset http_proxy
unset https_proxy
nnictl create --config config-dftuner1.yml -p 8130 --debug
sleep 3d
