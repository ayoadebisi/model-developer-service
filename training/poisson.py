import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pandas import DataFrame

from constants import ACTIVE_MODELS


def train_league_poisson(data):
    classification_features = build_nonsense(data)
    build_model(data, classification_features)


def build_model(features, classification_features):
    classification_outputs = get_classification_data(classification_features)
    home_data = process_features(features, classification_outputs, True)
    away_data = process_features(features, classification_outputs, False)
    input_data = pd.concat([home_data, away_data])
    formula = "goals ~ winner + team + opponent + home"
    model = smf.glm(formula=formula, data=input_data, family=sm.families.Poisson()).fit()

    print('Completed training poisson model')
    print(model.summary())

    ACTIVE_MODELS['poisson'][country.lower()] = model


def process_features(features, classification_outputs, home):
    input_data = {
        'winner': ['HomeWin' if np.argmax(row) == 1
                   else 'AwayWin' if np.argmax(row) == 2
                   else 'Tie'
                   for row in classification_outputs],
        'team': features['home_team'] if home else features['away_team'],
        'opponent': features['away_team'] if home else features['home_team'],
        'home': home,
        'goals': features['home_goal'] if home else features['away_goal']
    }

    return DataFrame(input_data)


def get_classification_data(classification_features):
    return ACTIVE_MODELS['classification'].predict(classification_features)


def build_nonsense(dataframe):
    data = {
        'performance': dataframe['h_pef_elo'] - dataframe['a_pef_elo'],
        'position': dataframe['home_pos'] - dataframe['away_pos'],
        'form': dataframe['h_form'] - dataframe['a_form'],
        'winning': dataframe['h_winning'] - dataframe['a_winning'],
        'unbeaten': dataframe['h_unbeaten'] - dataframe['a_unbeaten'],
        'clean_sheet': dataframe['h_clean_sheet'] - dataframe['a_clean_sheet'],
        'away': dataframe['a_away']
    }

    return DataFrame(data)
