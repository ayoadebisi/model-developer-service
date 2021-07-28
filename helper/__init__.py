import json
import pandas as pd

from pandas import DataFrame

from constants import TRAINING_DATA_PROVIDER_URL, COUNTRIES, TRAINING_DATA_COLUMNS, STAGE
from data.lambda_client_builder import LAMBDA_CLIENT
from training.classification import train_league_classification
from training.regression import train_league_regression


def obtain_training_data():
    try:
        training_data = []
        itr_len = len(COUNTRIES)

        if STAGE != 'DEVO':
            lambda_response = LAMBDA_CLIENT.invoke(
                FunctionName=TRAINING_DATA_PROVIDER_URL,
                InvocationType='RequestResponse'
            )['Payload'].read()
            lambda_response = json.loads(lambda_response)
            training_data = json.loads(lambda_response['body'])
        else:
            f = open('constants/training_data.json')
            training_data = json.load(f)
            itr_len = 1

        for i in range(itr_len):
            dataframe = DataFrame(training_data[i]['training_data'], columns=TRAINING_DATA_COLUMNS)
            dataframe.iloc[:, 2:] = dataframe.iloc[:, 2:].apply(pd.to_numeric)
            train_league_classification(dataframe, COUNTRIES[i].lower())
            train_league_regression(dataframe, COUNTRIES[i].lower())
    except Exception as e:
        print(f'Exception occurred whilst training model {e}')
