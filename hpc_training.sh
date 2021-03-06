#!/bin/sh
#SBATCH --account=def-ichiro
#SBATCH --output=logs/%N-%j.out

source /etc/bashrc
source calvo_env/bin/activate
# Run trainer
python run_calvo_trainer_mq.py "$@"
