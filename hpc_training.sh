#!/bin/sh
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%N-%j.out

source /etc/bashrc
source calvo_env/bin/activate
python run_calvo_trainer_mq.py "$@"
