import argparse
import base64
import os
import json
import requests
import pika


def download_resource(path, url, headers):
    r = requests.get(url, headers=headers, stream=True)
    if r.ok:
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)
        print(path + " is ok")
    else:
        print(path + " failed")

# Set up resources
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Path to input resource file")
parser.add_argument("callback_queue", help="RabbitMQ queue to reply on")
parser.add_argument("correlation_id", help="The correlation ID provided")

args = parser.parse_args()
input_file_path = args.input_file
response_queue = args.callback_queue
correlation_id = args.correlation_id

slurm_dir = os.environ["SLURM_TMPDIR"]
body = None
with open(input_file_path, 'r') as f:
    body = json.loads(f.read())
os.remove(input_file_path)

inputs = body['inputs']
settings = body['settings']

# Download Resources
print("Downloading resources...")
base_url = "http://" + os.environ["RABBITMQ_HOST"]
headers = {'Authorization': 'Token ' + settings['token']}

IMAGE_RES = os.path.join(slurm_dir, "image.png")
BG_RES = os.path.join(slurm_dir, "background.png")
MS_RES = os.path.join(slurm_dir, "music.png")
SL_RES = os.path.join(slurm_dir, "staff.png")
TL_RES = os.path.join(slurm_dir, "text.png")
SR_RES = os.path.join(slurm_dir, "regions.png")

BM_RES = os.path.join(slurm_dir, "background.hdf5")
MM_RES = os.path.join(slurm_dir, "music.hdf5")
SM_RES = os.path.join(slurm_dir, "staff.hdf5")
TM_RES = os.path.join(slurm_dir, "text.hdf5")

download_resource(IMAGE_RES, base_url + inputs['Image'], headers)
download_resource(BG_RES, base_url + inputs['Background'], headers)
download_resource(MS_RES, base_url + inputs['Music Layer'], headers)
download_resource(SL_RES, base_url + inputs['Staff Layer'], headers)
download_resource(TL_RES, base_url + inputs['Text'], headers)
download_resource(SR_RES, base_url + inputs['Selected Regions'], headers)

inputs = {
        "Image": IMAGE_RES,
        "Background": BG_RES,
        "Music Layer": MS_RES,
        "Staff Layer": SL_RES,
        "Text": TL_RES,
        "Selected Regions": SR_RES
        }
outputs = {
        "Background Model": BM_RES,
        "Music Symbol Model": MM_RES,
        "Staff Lines Model": SM_RES,
        "Text Model": TM_RES
        }

# Fast Trainer
print("Beginning fast trainer...")
from fast_calvo_trainer import FastCalvoTrainer

trainer = FastCalvoTrainer(inputs, settings, outputs)
trainer.run()

# Send response
print("Preparing response")
results = {}
with open(BM_RES, 'rb') as f:
    results['Background Model'] = base64.encodebytes(f.read()).decode('utf-8')
with open(MM_RES, 'rb') as f:
    results["Music Symbol Model"] = base64.encodebytes(f.read()).decode("utf-8")
with open(SM_RES, 'rb') as f:
    results["Staff Lines Model"] = base64.encodebytes(f.read()).decode("utf-8")
with open (TM_RES, 'rb') as f:
    results["Text Model"] = base64.encodebytes(f.read()).decode("utf-8")

body = json.dumps(results)

credentials = pika.PlainCredentials(os.environ["RABBITMQ_USER"], os.environ["RABBITMQ_PASSWORD"])
parameters = pika.ConnectionParameters(os.environ["RABBITMQ_HOST"], 5672, "/", credentials)
with pika.BlockingConnection(parameters) as conn:
    channel = conn.channel()
    channel.queue_declare(queue=response_queue)
    channel.basic_publish(
            exchange="",
            routing_key=response_queue,
            properties=pika.BasicProperties(correlation_id=correlation_id),
            body=body
        )
print("Done")
