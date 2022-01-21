import json
import pandas as pd

from pandas import DataFrame

from constants import TRAINING_DATA_COLUMNS, STAGE, LOCAL_TRAINING_DATA
from data.s3_client_builder import S3_CLIENT
from training.poisson import train_league_poisson

DATA_MEAN = {'data': pd.Series}
DATA_STD = {'data': pd.Series}


def obtain_training_data():
    try:
        if STAGE != 'DEVO':
            s3_response = S3_CLIENT['client'].get_object(
                Bucket='training-data-football-prediction',
                Key='training-data'
            )['Body'].read()
            training_data = json.loads(s3_response)
        else:
            f = open(LOCAL_TRAINING_DATA)
            training_data = json.load(f)

        frames = []

        for i in range(len(training_data)):
            frames.append(DataFrame(training_data[i]['training_data'], columns=TRAINING_DATA_COLUMNS))

        training_df = pd.concat(frames)

        train_league_poisson(training_df)
    except Exception as e:
        print(f'Exception occurred whilst training model {e}')
