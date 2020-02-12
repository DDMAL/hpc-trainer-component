# HPC Fast Trainer - HPC Component

This component runs the patchwise analysis training job and sends results back to [the Rodan task](https://github.com/JRegimbal/hpc-fast-trainer).
There are x major steps:

1. Connect to HPC-RabbitMQ and check if there is a job to run.
2. If there is a job, get the parameters for it and ACK the request.
This is to remove that entry from the queue and ensure another worker
won't attempt to perform the same task.
3. Authenticate with the Rodan API. This token with other parameters
is stored into a temporary file.
4. Start the [Slurm](https://slurm.schedmd.com/documentation.html) job. This runs on a separate node. From here, all
operations occur in the Slurm job once the scheduler starts it.
5. The resources are downloaded over HTTP from Rodan using the token obtained previously.
6. The Tensorflow job is run.
7. The resources are serialzed and converted to [base64](https://en.wikipedia.org/wiki/Base64).
8. The results are sent back to Rodan via a results queue and the Slurm job terminates.

## Quick Notes About Slurm

Slurm requires certain parameters to be specified before the job is scheduled.
These include the number of CPUs, number of GPUs, amount of memory, and total time
the job may run.
All of these are set via Rodan except for number of GPUs since currently the job
can only use one.

Note that allocating too little time/memory will result in the job failing. If that
happens the job on Rodan may hang indefinitely. Also note that allocating too many
resources will result in the scheduler taking a long time to actually start the
job.

## The Files

* `calvo_requirements.txt`: The Python dependencies for the trainer.
* `check.py`: Script that checks for a new job and submits a new Slurm job if one exists.
* `fast_calvo_trainer.py` and `training_engine_saw.py`: Actual patchwise trainer files.
* `hpc_training.sh`: Script defining the Slurm job and some batch parameters.
* `README.md`: This file!
* `requirements.txt`: Dependencies for `check.py`.
* `run_calvo_trainer_mq.py`: Script run in the Slurm worker that performs the training and submits the results.
* `run_check`: A file to be added to the crontab to intermittently run `check.py`.
