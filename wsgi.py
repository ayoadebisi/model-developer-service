import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import connexion
import sys


sys.path.append('/opt/python/current/app/model-developer-service/')

from connexion.resolver import RestyResolver
from data import lambda_client_builder, training_data_aggregator


def create_app():
    application = connexion.App(__name__, specification_dir='swagger/')
    application.add_api('ModelDeveloperService.yaml', resolver=RestyResolver('api'))
    lambda_client_builder.get()
    training_data_aggregator.get()
    return application


application = create_app()

if __name__ == '__main__':
    application.run()
