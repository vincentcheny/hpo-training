#!/bin/bash
#SBATCH --job-name=df
#SBATCH --mail-type=END, FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/output_bt_df.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=10000

#nnictl create --config config.yml -p 8081
#sleep 8h
#../../ngrok http 8081 -log=stdout &
#sleep 8h
python train-dragonfly.py