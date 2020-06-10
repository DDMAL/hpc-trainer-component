import json
import os
import pika
import logging
import ssl
import subprocess
import requests
import tempfile


logging.basicConfig(
    format='%(asctime)s %(message)s',
    filename='logs/check.log',
    level=logging.INFO
)

context = ssl.create_default_context()
ssl_options = pika.SSLOptions(context, os.environ['RABBITMQ_HOST'])
credentials = pika.PlainCredentials(
    os.environ['RABBITMQ_USER'],
    os.environ['RABBITMQ_PASSWORD']
)
parameters = pika.ConnectionParameters(
    host=os.environ['RABBITMQ_HOST'],
    port=5671,
    virtual_host='/',
    credentials=credentials,
    ssl_options=ssl_options
)

try:
    with pika.BlockingConnection(parameters) as conn:
        channel = conn.channel()
        channel.queue_declare(queue='hpc-jobs')
        # Check if there is an incoming job
        result = channel.basic_get('hpc-jobs')
        while result[0]:
            # Handle the incoming job in Slurm
            logging.info("Job received from queue")
            # Get settings for Slurm
            message = json.loads(result[2].decode('utf-8'))
            inputs = message['inputs']
            settings = message['settings']
            n_cpu = settings['CPUs']
            mem = settings['Maximum memory (MB)']
            time = settings['Maximum time (D-HH:MM)']
            mail = settings['Slurm Notification Email']

            # Authenticate User
            auth_url = "http://{}/api/auth/token/".format(os.environ["RODAN_HOST"])
            logging.info("Attempting to authenticate at {}...".format(auth_url))
            payload = {'username': os.environ['RODAN_USER'], 'password': os.environ['RODAN_PASSWORD']}
            response = requests.post(auth_url, data=payload)
            if not response.ok:
                logging.error("Bad response from server (" + response.url + ")")
                logging.error(response.text)
                quit()
            else:
                logging.info("Received code " + str(response.text) + " on authorization")

            settings['token'] = response.json()['token']
            logging.info("Token: " + settings['token'])
            message['settings'] = settings

            # Output the JSON body contents
            with tempfile.NamedTemporaryFile(dir=".", delete=False) as f:
                f.write(json.dumps(message).encode('utf-8'))
                run_array = [
                    'sbatch',
                    '--cpus-per-task='+str(n_cpu),
                    '--mem='+str(mem)+'M',
                    '--time='+str(time),
                    'hpc_training.sh',
                    f.name,
                    result[1].reply_to,
                    result[1].correlation_id
                ]
                logging.info("Reply queue: " + result[1].reply_to)
                if len(mail) > 0:
                    run_array.insert(1, '--mail-type=ALL')
                    run_array.insert(1, '--mail-user=' + mail)
                subprocess.run(run_array, check=True)
            channel.basic_ack(result[0].delivery_tag)
            result = channel.basic_get('hpc-jobs')  # Check for additional unscheduled jobs.
        else:
            logging.info("No job present.")
except pika.exceptions.AMQPConnectionError:
    logging.info("Could not connect.")
except Exception as e:
    logging.error("EXCEPTION")
    logging.error(e)
