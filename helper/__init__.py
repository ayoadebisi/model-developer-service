import json
import pandas as pd

from pandas import DataFrame

from constants import TRAINING_DATA_PROVIDER_URL, TRAINING_DATA_COLUMNS, STAGE
from data.lambda_client_builder import LAMBDA_CLIENT
from training.classification import train_league_classification
from training.regression import train_league_regression


def obtain_training_data():
    try:
        training_data = []

        if STAGE != 'DEVO':
            lambda_response = LAMBDA_CLIENT.invoke(
                FunctionName=TRAINING_DATA_PROVIDER_URL,
                InvocationType='RequestResponse'
            )['Payload'].read()
            lambda_response = json.loads(lambda_response)
            training_data = json.loads(lambda_response['body'])
        else:
            f = open('constants/europe_training_data.json')
            training_data = json.load(f)

        frames = []

        for i in range(len(training_data)):
            frames.append(DataFrame(training_data[i]['training_data'], columns=TRAINING_DATA_COLUMNS))

        training_df = pd.concat(frames)
        training_df.iloc[:, 2:] = training_df.iloc[:, 2:].apply(pd.to_numeric)

        train_league_classification(training_df)
        train_league_regression(training_df)
    except Exception as e:
        print(f'Exception occurred whilst training model {e}')
