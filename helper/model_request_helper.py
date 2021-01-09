from numpy import reshape, float64

from constants import CLASSIFICATION_INPUT_SHAPE, REGRESSION_INPUT_SHAPE
from helper.feature_calculator import get_elo_feature, get_standings_feature, get_form_feature


def build_classification_request(league_info, betting_info):
    classification_request = [get_elo_feature(league_info, 'offensive'), get_elo_feature(league_info, 'defensive'),
                              get_elo_feature(league_info, 'performance'),
                              get_standings_feature(league_info, 'pos'), get_form_feature(league_info, 'form'),
                              get_form_feature(league_info, 'winning'), get_form_feature(league_info, 'unbeaten'),
                              get_form_feature(league_info, 'home'), get_form_feature(league_info, 'away'),
                              betting_info['home_win'], betting_info['away_win'], betting_info['tie'],
                              betting_info['handicap']]

    return reshape(classification_request, (CLASSIFICATION_INPUT_SHAPE, 1)).T


def build_regression_request(league_info, betting_info, probabilities):
    regression_request = [get_standings_feature(league_info, 'gd'), get_form_feature(league_info, 'scoring'),
                          get_form_feature(league_info, 'clean_sheet'), betting_info['over'],
                          betting_info['under'], probabilities[0][1], probabilities[0][2], probabilities[0][0]]

    return reshape(regression_request, (REGRESSION_INPUT_SHAPE, 1)).T


def build_prediction_response(probabilities, goals):
    return {
            'forecast': {
                'home_win': round(float64(probabilities[0][1]), 3),
                'away_win': round(float64(probabilities[0][2]), 3),
                'tie': round(float64(probabilities[0][0]), 3)
            },
            'score': {
                'home': int(round(float64(goals[0][0]), 0)),
                'expected_home': round(float64(goals[0][0]), 2),
                'away': int(round(float64(goals[0][1]), 0)),
                'expected_away': round(float64(goals[0][1]), 2)
            }
        }
