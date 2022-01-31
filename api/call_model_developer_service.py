from flask import request

from constants import ACTIVE_MODELS
from helper.model_request_helper import build_prediction_response, build_poisson_prediction, build_poisson_distribution


class ModelDeveloperService(object):

    def get_prediction(self):
        try:
            league_info = request.json['league_info']

            print(f'Handling Request={league_info}')

            data = {
                'home_team': league_info['home_team'],
                'away_team': league_info['away_team']
            }

            poisson_model = ACTIVE_MODELS['poisson']

            print('Sending poisson inference request...')
            home_request, away_request = build_poisson_prediction(data)
            home_goals = poisson_model.predict(home_request)
            away_goals = poisson_model.predict(away_request)
            pred_distribution = build_poisson_distribution(home_goals, away_goals, max_goals=5)

            return build_prediction_response(home_goals.values[0], away_goals.values[0], pred_distribution)
        except Exception as e:
            print(f'Exception occurred whilst getting model prediction. {e}')
            return {
                "Error Message": e
            }, 500


class_instance = ModelDeveloperService()
