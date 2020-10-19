import argparse
import base64
import os
import logging
import json
import ssl
import sys
import requests
import pika


logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO
)


def send_to_rabbitmq(response_queue, correlation_id, body):
    logging.info("Response queue is (again): " + response_queue)
    context = ssl.create_default_context()
    ssl_options = pika.SSLOptions(context, os.environ["RABBITMQ_HOST"])
    credentials = pika.PlainCredentials(
        os.environ["RABBITMQ_USER"],
        os.environ["RABBITMQ_PASSWORD"]
    )
    parameters = pika.ConnectionParameters(
        os.environ["RABBITMQ_HOST"],
        5671,
        "/",
        credentials,
        ssl_options=ssl_options
    )
    with pika.BlockingConnection(parameters) as conn:
        channel = conn.channel()
        channel.queue_declare(queue=response_queue)
        channel.basic_publish(
                exchange="",
                routing_key=response_queue,
                properties=pika.BasicProperties(correlation_id=correlation_id),
                body=body
            )


def download_resource(path, url, headers):
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)
    logging.debug(path + " is ok")

# Set up resources
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input resource file")
    parser.add_argument("callback_queue", help="RabbitMQ queue to reply on")
    parser.add_argument("correlation_id", help="The correlation ID provided")

    args = parser.parse_args()
    input_file_path = args.input_file
    response_queue = args.callback_queue
    logging.info("Callback queue: " + response_queue)
    correlation_id = args.correlation_id

    slurm_dir = os.environ["SLURM_TMPDIR"]
    with open(input_file_path, 'r') as f:
        body = json.loads(f.read())
    os.remove(input_file_path)

    logging.info(body['settings'])

    # Download Resources
    base_url = "http://" + os.environ["RODAN_HOST"]
    headers = {'Authorization': 'Token ' + body['settings']['token']}
    logging.info("Downloading resources from " + base_url)

    # Handle reqired resources

    # Input
    IMAGE_RES = os.path.join(slurm_dir, "image.png")
    BG_RES = os.path.join(slurm_dir, "background.png")
    SR_RES = os.path.join(slurm_dir, "regions.png")

    # Output
    BM_RES = os.path.join(slurm_dir, "background.hdf5")

    # Download files from rodan
    download_resource(IMAGE_RES, base_url + body['inputs']['Image'], headers)
    download_resource(BG_RES, base_url + body['inputs']['Background'], headers)
    download_resource(SR_RES, base_url + body['inputs']['Selected Regions'], headers)

    inputs = {
        "Image": IMAGE_RES,
        "Background": BG_RES,
        "Selected Regions": SR_RES
    }
    outputs = {
        "Background Model": BM_RES,
    }

    # Handle Non-reqired resources
    # Optional named-layers (Input and download) "rgba PNG - Layer %d"
    for k in body["inputs"]:
        if k[:17] == 'rgba PNG - Layer ':
            l = k[17:18]
            layer_file = "layer_%d.png" % l
            model_file = "model_%d.png" % l

            # Declare input filepath
            inputs['rgba PNG - Layer %d' % l] = os.path.join(slurm_dir, layer_file)

            # Download File
            download_resource(layer_file, base_url + body["inputs"]["rgba PNG - Layer %d" % l], headers)

            # Declare output filepath
            outputs['Model %d' % l] = os.path.join(slurm_dir, model_file)

    # Fast Trainer
    logging.info("Beginning fast trainer...")
    from fast_calvo_trainer import FastCalvoTrainer

    trainer = FastCalvoTrainer(inputs, body['settings'], outputs)
    trainer.run()

    # Send response
    logging.info("Preparing response")
    results = {}
    with open(BM_RES, 'rb') as f:
        results['Background Model'] = base64.encodebytes(f.read()).decode('utf-8')

    # Send response (for Optional layers) "Model %d"
    for k in body["outputs"]:
        if k[:5] == "Model":
            l = k[6:7]
            model = 'Model %d' % l
            with open(outputs[model], 'rb') as f:
                results[model] = base64.encodebytes(f.read()).decode("utf-8")

    body = json.dumps(results)

except Exception as e:
    # We need to reply with something since the initial message was ACKed
    body = json.dumps({'error': str(e)})
    logging.error(e)
finally:
    send_to_rabbitmq(response_queue, correlation_id, body)
logging.info("Done")
