#!/bin/sh
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --output=%N-%j.out

module load python/3.7.4
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install -r calvo_requirements.txt
python run_calvo_trainer_mq.py "$@"
