from flask import request
from numpy import reshape, round, float64

from constants import BEST_RATED_MODELS, ACTIVE_MODELS
from helper.feature_calculator import get_elo_feature, get_standings_feature, get_form_feature, get_offense_differential


class ModelDeveloperService(object):

    def get_prediction(self):
        league_info = request.json['predictionRequest']['league_info']
        betting_info = request.json['predictionRequest']['betting_info']
        classification_model = ACTIVE_MODELS['classification'][league_info['country']]
        regression_model = ACTIVE_MODELS['regression'][league_info['country']]

        classification_request = [get_elo_feature(league_info, 'offensive'), get_elo_feature(league_info, 'defensive'),
                                  get_elo_feature(league_info, 'performance'),
                                  get_standings_feature(league_info, 'pos'), get_form_feature(league_info, 'form'),
                                  get_form_feature(league_info, 'winning'), get_form_feature(league_info, 'unbeaten'),
                                  get_form_feature(league_info, 'home'), get_form_feature(league_info, 'away'),
                                  betting_info['home_win'], betting_info['away_win'], betting_info['tie'],
                                  betting_info['handicap']]

        regression_request = [get_offense_differential(league_info, True),
                              get_offense_differential(league_info, False),
                              get_elo_feature(league_info, 'performance'),
                              get_standings_feature(league_info, 'gd'),
                              get_form_feature(league_info, 'scoring'), get_form_feature(league_info, 'clean_sheet'),
                              betting_info['over'], betting_info['under'], betting_info['handicap']]

        classification_request = reshape(classification_request, (13, 1)).T
        regression_request = reshape(regression_request, (9, 1)).T
        probabilities = classification_model.predict(classification_request)
        goals = round(regression_model.predict(regression_request), 0)

        response = {
            'forecast': {
                'home_win': float64(probabilities[0][1]),
                'away_win': float64(probabilities[0][2]),
                'tie': float64(probabilities[0][0])
            },
            'score': {
                'home': float64(goals[0][0]),
                'away': float64(goals[0][1])
            }
        }

        return response

    def get_ratings(self):
        return BEST_RATED_MODELS


class_instance = ModelDeveloperService()
