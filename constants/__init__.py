STAGE = 'DEVO'

SEED = 78
TEST_SIZE = 0.2
SHUFFLE = False
EPOCHS = 50
LEARNING_RATE = 0.001
DECAY_RATE = LEARNING_RATE / EPOCHS
BATCH_SIZE = 128
VERBOSE = 0

CLASSIFICATION_INPUT_SHAPE = 13
REGRESSION_INPUT_SHAPE = 8

DATA_PROVIDER_URL = 'https://y95xe287sc.execute-api.us-east-1.amazonaws.com/Prod/provider/'
TRAINING_DATA_PROVIDER_URL = 'arn:aws:lambda:us-east-1:366364139691:function:Football-Data-Aggregator-' \
                             'DataAggregatorLambda-JMAYFOUQRQR4'

COUNTRIES = ['england', 'spain', 'france', 'italy', 'germany']

TRAINING_DATA_COLUMNS = ['home_team', 'away_team', 'home_goal', 'away_goal', 'outcome', 'h_off_elo', 'h_def_elo',
                         'h_pef_elo', 'a_off_elo', 'a_def_elo', 'a_pef_elo', 'home_pos', 'home_gd', 'home_pts',
                         'away_pos', 'away_gd', 'away_pts', 'h_form', 'h_winning', 'h_unbeaten', 'h_home', 'h_away',
                         'h_clean_sheet', 'h_scoring', 'a_form', 'a_winning', 'a_unbeaten', 'a_home', 'a_away',
                         'a_clean_sheet', 'a_scoring', 'home_odds', 'draw_odds', 'away_odds', 'over', 'under',
                         'handicap']
CLASSIFICATION_TRAINING_FEATURE_COLUMNS = ['h_off_elo', 'h_def_elo', 'h_pef_elo', 'a_off_elo', 'a_def_elo', 'a_pef_elo',
                                           'h_form', 'h_winning', 'h_unbeaten', 'h_home', 'h_away', 'a_form',
                                           'a_winning', 'a_unbeaten', 'a_home', 'a_away', 'home_pos', 'away_pos',
                                           'home_odds', 'draw_odds', 'away_odds', 'handicap']
REGRESSION_TRAINING_FEATURE_COLUMNS = ['h_scoring', 'h_clean_sheet', 'a_scoring', 'a_clean_sheet', 'home_gd', 'away_gd',
                                       'over', 'under']

BEST_RATED_MODELS = {
    'classification': {'england': 0.0, 'spain': 0.0, 'germany': 0.0, 'italy': 0.0, 'france': 0.0},
    'regression': {'england': 0.0, 'spain': 0.0, 'germany': 0.0, 'italy': 0.0, 'france': 0.0}
}

ACTIVE_MODELS = {
    'classification': {'england': None, 'spain': None, 'germany': None, 'italy': None, 'france': None},
    'regression': {'england': None, 'spain': None, 'germany': None, 'italy': None, 'france': None}
}
