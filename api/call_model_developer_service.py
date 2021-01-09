from flask import request
from numpy import round

from constants import BEST_RATED_MODELS, ACTIVE_MODELS
from helper.model_request_helper import build_classification_request, build_regression_request, \
    build_prediction_response


class ModelDeveloperService(object):

    def get_prediction(self):
        league_info = request.json['predictionRequest']['league_info']
        betting_info = request.json['predictionRequest']['betting_info']

        classification_model = ACTIVE_MODELS['classification'][league_info['country']]
        regression_model = ACTIVE_MODELS['regression'][league_info['country']]

        classification_request = build_classification_request(league_info, betting_info)
        regression_request = build_regression_request(league_info, betting_info)

        probabilities = classification_model.predict(classification_request)
        goals = round(regression_model.predict(regression_request), 0)

        return build_prediction_response(probabilities, goals)

    def get_ratings(self):
        return BEST_RATED_MODELS


class_instance = ModelDeveloperService()
