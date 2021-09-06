import numpy as np

from pandas import DataFrame
from numpy import reshape, float64

from training.regression import NUM_TEAMS, hash_team_name


def build_classification_request(request_data):
    classification_request = [request_data['PerformanceElo'], request_data['Position'], request_data['Form'],
                              request_data['WinningStreak'], request_data['UnbeatenStreak'], request_data['CleanSheet'],
                              request_data['HeadToHeadCS'], request_data['HeadToHeadForm'],
                              request_data['HeadToHeadGoal'], request_data['HeadToHeadGoalAvg'],
                              request_data['HeadToHeadScoring'], request_data['HeadToHeadUnbeaten'],
                              request_data['HeadToHeadWinning'], request_data['HeadToHeadWins'],
                              request_data['AwayForm']]

    return reshape(classification_request, (len(classification_request), 1)).T


def build_regression_request(request_data, probabilities):
    home_team = hash_team_name(request_data['HomeTeam'].replace(" ", ""), NUM_TEAMS['Length'])
    away_team = hash_team_name(request_data['AwayTeam'].replace(" ", ""), NUM_TEAMS['Length'])
    regression_request = [probabilities[0][1], probabilities[0][2], probabilities[0][0], request_data['AwayCleanSheet'],
                          request_data['HeadToHeadGoalAvg'], request_data['HeadToHeadUnbeaten'],
                          request_data['HeadToHeadWinning'], request_data['HeadToHeadWins'],
                          home_team, away_team]

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
                'home_win': round(float64(probabilities[0][1]), 3),
                'away_win': round(float64(probabilities[0][2]), 3),
                'tie': round(float64(probabilities[0][0]), 3)
            },
            'score': {
                'home': int(round(float64(home_goals), 0)),
                'expected_home': round(float64(home_goals), 2),
                'away': int(round(float64(away_goals), 0)),
                'expected_away': round(float64(away_goals), 2)
            }
        }
