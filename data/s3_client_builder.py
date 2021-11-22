import datetime
import os
import threading
import boto3

from botocore.config import Config

S3_CLIENT = {'client': None}


def get():
    global S3_CLIENT
    print('Renewing S3 Client Credentials ', datetime.datetime.now())
    S3_CLIENT['client'] = boto3.client(
        's3',
        aws_access_key_id=os.environ['ACCESS_KEY'],
        aws_secret_access_key=os.environ['SECRET_KEY'],
        config=Config(
            read_timeout=900,
            connect_timeout=900,
            retries={"max_attempts": 0}
        )
    )
    print('Renewed S3 Client Credentials', datetime.datetime.now())
    threading.Timer(3600, get).start()
