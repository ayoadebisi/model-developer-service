import json
import pandas as pd

from pandas import DataFrame
from matplotlib import pyplot

from constants import TRAINING_DATA_COLUMNS, STAGE, LOCAL_TRAINING_DATA
from data.s3_client_builder import S3_CLIENT
from training.classification import train_league_classification
from training.regression import train_league_regression

DATA_MEAN = {'data': pd.Series}
DATA_STD = {'data': pd.Series}


def obtain_training_data():
    try:
        if STAGE != 'DEVO':
            s3_response = S3_CLIENT['client'].get_object(
                Bucket='training-data-football-prediction',
                Key='training-data'
            )['Body'].read()
            training_data = json.loads(s3_response)
        else:
            f = open(LOCAL_TRAINING_DATA)
            training_data = json.load(f)

        frames = []

        for i in range(len(training_data)):
            frames.append(DataFrame(training_data[i]['training_data'], columns=TRAINING_DATA_COLUMNS))

        training_df = pd.concat(frames)

        transformed_df = process_data(training_df)

        global DATA_MEAN, DATA_STD
        DATA_MEAN['data'] = transformed_df.mean()
        DATA_STD['data'] = transformed_df.std()

        normalized_data = standardized_data(transformed_df)
        normalized_data[['home_goal', 'away_goal', 'outcome']] = training_df[['home_goal', 'away_goal', 'outcome']]
        display_corr(normalized_data)

        train_league_classification(normalized_data)
        train_league_regression(normalized_data)
    except Exception as e:
        print(f'Exception occurred whilst training model {e}')


def display_hist(training_data):
    training_data.hist(bins=30, figsize=(15, 10), grid=False)
    pyplot.tight_layout()
    pyplot.show()


def standardized_data(training_data):
    return (training_data - training_data.mean()) / training_data.std()


def display_corr(training_data):
    home_corr = training_data.corr()['home_goal'].sort_values()
    away_corr = training_data.corr()['away_goal'].sort_values()
    outcome_corr = training_data.corr()['outcome'].sort_values()
    print(f'Correlation for Home Goals: {home_corr}')
    print(f'Correlation for Away Goals: {away_corr}')
    print(f'Correlation for Outcome: {outcome_corr}')


def process_data(dataframe):
    data = {
        'home_goal': dataframe['home_goal'],
        'away_goal': dataframe['away_goal'],
        'outcome': dataframe['outcome'],
        'offense': dataframe['h_off_elo'] - dataframe['a_off_elo'],
        'defense': dataframe['h_def_elo'] - dataframe['a_def_elo'],
        'performance': dataframe['h_pef_elo'] - dataframe['a_pef_elo'],
        'position': dataframe['home_pos'] - dataframe['away_pos'],
        'goal_difference': dataframe['home_gd'] - dataframe['away_gd'],
        'points': dataframe['home_pts'] - dataframe['away_pts'],
        'form': dataframe['h_form'] - dataframe['a_form'],
        'winning': dataframe['h_winning'] - dataframe['a_winning'],
        'unbeaten': dataframe['h_unbeaten'] - dataframe['a_unbeaten'],
        'home_form': dataframe['h_home'] - dataframe['a_home'],
        'away_form': dataframe['h_away'] - dataframe['a_away'],
        'clean_sheet': dataframe['h_clean_sheet'] - dataframe['a_clean_sheet'],
        'scoring_streak': dataframe['h_scoring'] - dataframe['a_scoring'],
        'head_to_head_cs': dataframe['head_to_head_clean_sheet_1'] - dataframe['head_to_head_clean_sheet_2'],
        'head_to_head_form': dataframe['head_to_head_form_1'] - dataframe['head_to_head_form_2'],
        'head_to_head_goal': dataframe['head_to_head_goal_1'] - dataframe['head_to_head_goal_2'],
        'head_to_head_goal_avg': dataframe['head_to_head_goal_avg_1'] - dataframe['head_to_head_goal_avg_2'],
        'head_to_head_scoring': dataframe['head_to_head_scoring_1'] - dataframe['head_to_head_scoring_2'],
        'head_to_head_unbeaten': dataframe['head_to_head_unbeaten_1'] - dataframe['head_to_head_unbeaten_2'],
        'head_to_head_winning': dataframe['head_to_head_winning_1'] - dataframe['head_to_head_winning_2'],
        'head_to_head_wins': dataframe['head_to_head_wins_1'] - dataframe['head_to_head_wins_2'],
        'home_odds': dataframe['home_odds'],
        'draw_odds': dataframe['home_odds'],
        'away_odds': dataframe['home_odds'],
        'over': dataframe['home_odds'],
        'under': dataframe['home_odds'],
        'handicap': dataframe['handicap']
    }

    return DataFrame(data)
