import numpy as np

from pandas import DataFrame
from numpy import reshape, float64

from helper import DATA_MEAN, DATA_STD
from constants import NORMALIZATION_KEYS
from training.classification import get_regression_features


def normalize_data(data, key):
    return (data - DATA_MEAN['data'][NORMALIZATION_KEYS[key]]) / DATA_STD['data'][NORMALIZATION_KEYS[key]]


def get_request_features(request_data, feature_keys):
    return [normalize_data(request_data[key], key) for key in feature_keys]


def build_classification_request(request_data):
    classification_request = get_request_features(request_data, NORMALIZATION_KEYS)
    return reshape(classification_request, (len(classification_request), 1)).T


def build_regression_request(request_data, probabilities):
    probabilities_list = [probabilities[0][2], probabilities[0][0], probabilities[0][1]]
    feature_keys = get_regression_features()
    regression_request = get_request_features(request_data, feature_keys) + probabilities_list
    return reshape(regression_request, (len(regression_request), 1)).T


def build_poisson_request(request_data, probabilities, home, multiplier):
    poisson_request = [[request_data['OffensiveElo'] * multiplier, request_data['DefensiveElo'] * multiplier,
                        request_data['PerformanceElo'] * multiplier, request_data['Position'] * multiplier,
                        request_data['Form'] * multiplier,
                        request_data['WinningStreak'] * multiplier, request_data['UnbeatenStreak'] * multiplier,
                        request_data['HomeForm'] * multiplier,
                        request_data['AwayForm'] * multiplier, request_data['HomeWin'], request_data['AwayWin'],
                        request_data['Draw'], request_data['AsianHandicap'],
                        request_data['GoalDifference'] * multiplier, request_data['ScoringStreak'] * multiplier,
                        request_data['CleanSheet'] * multiplier, request_data['Over'],
                        np.argmax(probabilities[0]), home]]

    return DataFrame(poisson_request, columns=parse_formula_to_list())


def parse_formula_to_list():
    formula = "offense + defense + performance + position + form + winning_streak + unbeaten_streak + "\
              "home_form + away_form + home_odds + away_odds + draw_odds + handicap + "\
              "goal_difference + scoring_streak + clean_sheet_streak + over + winner + home"
    return [x.strip() for x in formula.split("+")]


def build_default_response():
    return {
        'forecast': {
            'home_win': 0.33,
            'away_win': 0.33,
            'tie': 0.34
        },
        'score': {
            'home': 0,
            'expected_home': 0.0,
            'away': 0,
            'expected_away': 0.0
        }
    }


def build_prediction_response(probabilities, home_goals, away_goals):
    return {
            'forecast': {
                'home_win': round(probabilities[0][2], 3),
                'away_win': round(probabilities[0][0], 3),
                'tie': round(probabilities[0][1], 3)
            },
            'score': {
                'home': int(round(negative_goal_check(float64(home_goals)), 0)),
                'expected_home': round(negative_goal_check(float64(home_goals)), 2),
                'away': int(round(negative_goal_check(float64(away_goals)), 0)),
                'expected_away': round(negative_goal_check(float64(away_goals)), 2)
            }
        }


def negative_goal_check(goal):
    return 0 if goal < 0 else goal
