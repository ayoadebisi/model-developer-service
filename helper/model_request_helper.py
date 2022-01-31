import numpy as np

from pandas import DataFrame
from scipy.stats import poisson
from numpy import float64


def build_poisson_prediction(request_data):
    home_poisson_request = {'team': request_data['home_team'], 'opponent': request_data['away_team'], 'home': True}
    away_poisson_request = {'team': request_data['away_team'], 'opponent': request_data['home_team'], 'home': False}

    return DataFrame(data=home_poisson_request, index=[1]), DataFrame(data=away_poisson_request, index=[1])


def build_poisson_distribution(home_goals, away_goals, max_goals=10):
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                 [home_goals.values[0], away_goals.values[0]]]
    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))


def build_prediction_response(home_goals, away_goals, probabilities):
    home_win = np.sum(np.tril(probabilities, -1))
    away_win = np.sum(np.triu(probabilities, 1))
    tie = np.sum(np.diag(probabilities))
    score = get_score_tuple(probabilities)
    return {
            'forecast': {
                'home_win': round(home_win, 3),
                'away_win': round(away_win, 3),
                'tie': round(tie, 3)
            },
            'score': {
                'model_home': int(round(negative_goal_check(float64(home_goals)), 0)),
                'model_away': int(round(negative_goal_check(float64(away_goals)), 0)),
                'matrix_home': int(score[0]),
                'matrix_away': int(score[1]),
                'expected_home': round(negative_goal_check(float64(home_goals)), 2),
                'expected_away': round(negative_goal_check(float64(away_goals)), 2)
            }
        }


def get_score_tuple(probabilities):
    max_index = probabilities.argmax()
    return np.unravel_index(max_index, probabilities.shape)


def negative_goal_check(goal):
    return 0 if goal < 0 else goal
