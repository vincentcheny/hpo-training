#!/bin/bash
#SBATCH --job-name=dftuner_inception
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/research/dept7/ychen/model/hpo-training/inception10G-tf2/slurm-log/4-DFTuner.log
#SBATCH --gres=gpu:2
#SBATCH -w gpu19
#SBATCH -p batch_72h

unset http_proxy
unset https_proxy
nnictl create --config config-dftuner.yml -p 8130 --debug
sleep 3d
