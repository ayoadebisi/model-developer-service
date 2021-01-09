TIMEOUT = 86400

SEASON = 2021

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

DATA_PROVIDER_URL = 'http://soccer-predictor-4.eba-kcmdarmk.us-east-2.elasticbeanstalk.com'
DATA_PROVIDER_ENDPOINT = '/v1/ResultsProcessorService/'
TRAINING_DATA_ENDPOINT = '/v1/ResultsProcessorService/training/'
FORM_DATA_ENDPOINT = '/v1/ResultsProcessorService/form/'

COUNTRIES = ['england']  # ['england', 'spain', 'france', 'italy', 'germany']
RATING_TYPES = ['performance', 'offensive', 'defensive']

CLASSIFICATION_TRAINING_FEATURE_COLUMNS = ['h_off_elo', 'h_def_elo', 'h_pef_elo', 'a_off_elo', 'a_def_elo', 'a_pef_elo',
                                           'h_form', 'h_winning', 'h_unbeaten', 'h_home', 'h_away', 'a_form',
                                           'a_winning', 'a_unbeaten', 'a_home', 'a_away', 'home_pos', 'away_pos',
                                           'home_odds', 'draw_odds', 'away_odds', 'handicap']

REGRESSION_TRAINING_FEATURE_COLUMNS = ['h_scoring', 'h_clean_sheet', 'a_scoring', 'a_clean_sheet', 'home_gd', 'away_gd',
                                       'over', 'under']

LEAGUE_MAP = {
    'england': 'epl',
    'spain': 'la_liga',
    'germany': 'bundesliga',
    'italy': 'serie_a',
    'france': 'ligue_1',
}

BEST_RATED_MODELS = {
    'classification': {'england': 0.0, 'spain': 0.0, 'germany': 0.0, 'italy': 0.0, 'france': 0.0},
    'regression': {'england': 0.0, 'spain': 0.0, 'germany': 0.0, 'italy': 0.0, 'france': 0.0}
}

ACTIVE_MODELS = {
    'classification': {'england': None, 'spain': None, 'germany': None, 'italy': None, 'france': None},
    'regression': {'england': None, 'spain': None, 'germany': None, 'italy': None, 'france': None}
}

LEAGUE_STANDINGS = {
    'england': None, 'spain': None, 'germany': None, 'italy': None, 'france': None
}

FORM_DATA = {
    'england': None, 'spain': None, 'germany': None, 'italy': None, 'france': None
}

ELO_DATA = {
    'england': None, 'spain': None, 'germany': None, 'italy': None, 'france': None
}
