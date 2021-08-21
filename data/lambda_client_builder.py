import datetime
import os
import threading
import boto3

from botocore.config import Config

LAMBDA_CLIENT = None


def get():
    global LAMBDA_CLIENT
    print('Renewing Lambda Client Credentials ', datetime.datetime.now())
    LAMBDA_CLIENT = boto3.client(
        'lambda',
        aws_access_key_id=os.environ['ACCESS_KEY'],
        aws_secret_access_key=os.environ['SECRET_KEY'],
        config=Config(
            read_timeout=900,
            connect_timeout=900,
            retries={"max_attempts": 0}
        )
    )
    print('Renewed Lambda Client Credentials', datetime.datetime.now())
    threading.Timer(3600, get).start()