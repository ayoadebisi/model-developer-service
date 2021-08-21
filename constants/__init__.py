import os

STAGE = os.environ['STAGE']

SEED = int(os.environ['SEED'])
TEST_SIZE = float(os.environ['TEST_SIZE'])
SHUFFLE = os.environ['SHUFFLE']
EPOCHS = int(os.environ['EPOCHS'])
LEARNING_RATE = float(os.environ['LEARNING_RATE'])
DECAY_RATE = LEARNING_RATE / EPOCHS
BATCH_SIZE = int(os.environ['BATCH_SIZE'])
VERBOSE = int(os.environ['VERBOSE'])

DATA_PROVIDER_URL = os.environ['DATA_PROVIDER_URL']
TRAINING_DATA_PROVIDER_URL = os.environ['TRAINING_DATA_PROVIDER_URL']

TRAINING_DATA_COLUMNS = ['home_team', 'away_team', 'home_goal', 'away_goal', 'outcome', 'h_off_elo', 'h_def_elo',
                         'h_pef_elo', 'a_off_elo', 'a_def_elo', 'a_pef_elo', 'home_pos', 'home_gd', 'home_pts',
                         'away_pos', 'away_gd', 'away_pts', 'h_form', 'h_winning', 'h_unbeaten', 'h_home', 'h_away',
                         'h_clean_sheet', 'h_scoring', 'a_form', 'a_winning', 'a_unbeaten', 'a_home', 'a_away',
                         'a_clean_sheet', 'a_scoring', 'home_odds', 'draw_odds', 'away_odds', 'over', 'under',
                         'handicap']

BEST_RATED_MODELS = {
    'classification': 0.0,
    'regression': 0.0
}

ACTIVE_MODELS = {
    'classification': None,
    'regression': None,
    'poisson': None
}
