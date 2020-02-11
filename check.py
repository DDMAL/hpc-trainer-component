import json
import os
import pika
import subprocess
import requests
import tempfile

credentials = pika.PlainCredentials(os.environ['RABBITMQ_USER'], os.environ['RABBITMQ_PASSWORD'])
parameters = pika.ConnectionParameters(host=os.environ['RABBITMQ_HOST'], port=5672, virtual_host='/', credentials=credentials)

try:
    with pika.BlockingConnection(parameters) as conn:
        channel = conn.channel()
        channel.queue_declare(queue='hpc-jobs')
        # Check if there is an incoming job
        result = channel.basic_get('hpc-jobs')
#        method_frame, header_frame, body = channel.basic_get('hpc-jobs')
        if result[0]:
            # Handle the incoming job in Slurm
            print("Job received from queue")
            # Get settings for Slurm
            message = json.loads(result[2].decode('utf-8'))
            inputs = message['inputs']
            settings = message['settings']
            n_cpu = settings['CPUs']
            mem = settings['Maximum memory (MB)']
            time = settings['Maximum time (D-HH:MM)']

            # Authenticate User
            payload = {'username': os.environ['RODAN_USER'], 'password': os.environ['RODAN_PASSWORD']}
            response = requests.post('http://' + os.environ['RODAN_HOST'] + '/auth/token/', data=payload)
            if not response.ok:
                print("Error > Bad response from server (" + response.url + ")")
                print("      > " + response.text)
                quit()

            settings['token'] = response.json()['token']
            message['settings'] = settings 

            # Output the JSON body contents
            with tempfile.NamedTemporaryFile(dir=".", delete=False) as f:
                f.write(json.dumps(message).encode('utf-8'))
                subprocess.run([
                    'sbatch',
                    '--cpus-per-task='+str(n_cpu),
                    '--mem='+str(mem)+'M',
                    '--time='+str(time),
                    'hpc_training.sh',
                    f.name,
                    result[1].reply_to,
                    result[1].correlation_id
                ])
            channel.basic_ack(result[0].delivery_tag)
except pika.exceptions.AMQPConnectionError:
    pass
