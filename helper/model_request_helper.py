from numpy import reshape, float64

from constants import CLASSIFICATION_INPUT_SHAPE, REGRESSION_INPUT_SHAPE


def build_classification_request(request_data):
    classification_request = [request_data['OffensiveElo'], request_data['DefensiveElo'],
                              request_data['PerformanceElo'], request_data['Position'], request_data['Form'],
                              request_data['WinningStreak'], request_data['UnbeatenStreak'], request_data['HomeForm'],
                              request_data['AwayForm'], request_data['HomeWin'], request_data['AwayWin'],
                              request_data['Draw'], request_data['AsianHandicap']]

    return reshape(classification_request, (CLASSIFICATION_INPUT_SHAPE, 1)).T


def build_regression_request(request_data, probabilities):
    regression_request = [request_data['GoalDifference'], request_data['ScoringStreak'], request_data['CleanSheet'],
                          request_data['Over'], request_data['Under'], probabilities[0][1], probabilities[0][2],
                          probabilities[0][0]]

    return reshape(regression_request, (REGRESSION_INPUT_SHAPE, 1)).T


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
