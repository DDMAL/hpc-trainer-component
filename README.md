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
* `credentials.env`: File containing the credentials to log into Rodan and RabbitMQ.
*You must fill this file out!*
* `fast_calvo_trainer.py` and `training_engine_saw.py`: Actual patchwise trainer files.
* `hpc_training.sh`: Script defining the Slurm job and some batch parameters.
* `README.md`: This file!
* `requirements.txt`: Dependencies for `check.py`.
* `run_calvo_trainer_mq.py`: Script run in the Slurm worker that performs the training and submits the results.
* `run_check`: A file to be added to the crontab to intermittently run `check.py`.

# Set Up

1. Clone this repository [somewhere where jobs can be scheduled](https://docs.computecanada.ca/wiki/Running_jobs#Cluster_particularities).
This is either `scratch` or a project directory on Cedar.
2. Change directory to this repository's contents, which will now just be called `{path}`.
3. Set up a virtual environment for Python 3.7 and install the necessary dependencies.
    ```bash
    module load python/3.7
    virtualenv env
    source env/bin/activate
    pip install -r requirements.txt
    ```
    This environment is activated by [run_check](./run_check).
4. Set up a second virtual environment for the calvo code, also using Python 3.7.
    ```bash
    module load python/3.7
    virtualenv calvo_env
    source calvo_env/bin/activate
    pip install -r calvo_requirements.txt
    ```
    This environment gets activated by [hpc_training.sh](./hpc_training.sh).
5. Export the proper environment variables in `credentials.env`. This *must* include:
   * `RABBITMQ_USER`, the user account for RabbitMQ.
   * `RABBITMQ_PASSWORD`, the corresponding password for RabbitMQ.
   * `RABBITMQ_HOST`, the IP address or host where RabbitMQ is on port 5671.
   * `RODAN_USER`, the user account, with enough permissions, for Rodan.
   * `RODAN_PASSWORD`, the corresponding password for Rodan.
   * `RODAN_HOST`, the IP address or host where the Rodan API is on port 80. 

    Make sure to add `export` to the start of each line with a variable. The credentials for `RABBITMQ_USER` and `RABBITMQ_PASSWORD` can be found in the [rodan-docker (private) repository](https://github.com/DDMAL/rodan-docker/tree/master/hpc-rabbitmq/scripts). For the host in `RABBITMQ_HOST` and `RODAN_HOST`, use `rodan2.simssa.ca`. For rodan's host and password, make sure to use the credentials of a user with enough permissions.

6. Add `run_check` to your crontab. For example to check for jobs every hour on the hour and log to a file
called `logs/run_check.log`, run `crontab -e` and add the following line:
    ```sh
    0 * * * * SSL_CERT_FILE={location} {path}/run_check >> {path}/logs/run_check.log 2>&1
    ```
    * `{location}` = location of the SSL certificate, which can be aquired by `echo $SSL_CERT_FILE`.
    * `{path}` = hpc-trainer-component directory.

    For example, the entire command may look like: `0 * * * * SSL_CERT_FILE='/etc/pki/tls/certs/ca-bundle.crt'  ~/scratch/hpc-trainer-component/run_check >> ~/scratch/hpc-trainer-component/logs/run_check.log 2>&1`
