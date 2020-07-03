import argparse
import logging
import json
import ssl
import os
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

parser = argparse.ArgumentParser()
parser.add_argument("callback_queue", help="RabbitMQ queue to reply on")
parser.add_argument("correlation_id", help="The correlation ID provided")

args = parser.parse_args()
response_queue = args.callback_queue
logging.info("Callback queue: " + response_queue)
correlation_id = args.correlation_id

send_to_rabbitmq(response_queue, correlation_id, json.dumps({"error": "a slurm error occurred"}))
