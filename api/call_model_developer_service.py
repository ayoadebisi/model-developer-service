import requests
import json

from flask import request

from constants import BEST_RATED_MODELS, ACTIVE_MODELS, DATA_PROVIDER_URL
from helper.model_request_helper import build_classification_request, build_prediction_response, \
    build_default_response, build_regression_request


class ModelDeveloperService(object):

    def get_prediction(self):
        try:
            league_info = request.json['predictionRequest']['league_info']

            print(f'Handling Request={league_info}')

            data = {
                'home_team': league_info['home_team'],
                'away_team': league_info['away_team']
            }

            prediction_data = json.loads(requests.post(url=DATA_PROVIDER_URL, data=json.dumps(data)).text)

            print(f'Prediction Data={prediction_data}')

            if not bool(prediction_data):
                print('Prediction data is empty, returning default response.')
                return build_default_response()

            classification_model = ACTIVE_MODELS['classification']
            regression_model = ACTIVE_MODELS['regression']

            if not (classification_model or regression_model):
                return {
                    'message': 'Internal Error: Prediction model isn\'t ready. Try again later'
                }, 500

            print('Sending classification inference request...')
            classification_request = build_classification_request(prediction_data)
            probabilities = classification_model.predict(classification_request)

            print('Sending regression inference request...')
            regression_request = build_regression_request(prediction_data, probabilities)
            goals = regression_model.predict(regression_request)

            return build_prediction_response(probabilities, goals[0][0], goals[0][1])
        except Exception as e:
            print(f'Exception occurred whilst getting model prediction. {e}')
            return {
                "Error Message": e
            }, 500

    def get_ratings(self):
        return BEST_RATED_MODELS


class_instance = ModelDeveloperService()
