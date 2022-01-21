import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pandas import DataFrame

from constants import ACTIVE_MODELS


def train_league_poisson(data):
    build_model(data)


def build_model(features):
    home_data = process_features(features, True)
    away_data = process_features(features, False)
    input_data = pd.concat([home_data, away_data])
    formula = "goals ~ team + opponent + home"
    model = smf.glm(formula=formula, data=input_data, family=sm.families.Poisson()).fit()

    print('Completed training poisson model')
    print(model.summary())

    ACTIVE_MODELS['poisson'] = model


def process_features(features, home):
    input_data = {
        'team': features['home_team'] if home else features['away_team'],
        'opponent': features['away_team'] if home else features['home_team'],
        'home': home,
        'goals': features['home_goal'] if home else features['away_goal']
    }

    return DataFrame(input_data)
