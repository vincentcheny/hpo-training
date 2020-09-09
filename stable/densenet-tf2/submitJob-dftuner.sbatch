#!/bin/bash
#SBATCH --job-name=blind_dftuner
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/research/dept7/ychen/model/blind/slurm-log/3-DFTuner.log
#SBATCH --gres=gpu:2
#SBATCH -w gpu17
#SBATCH -p batch_72h

unset http_proxy
unset https_proxy
nnictl create --config config-dftuner.yml -p 8134 --debug
sleep 3d
