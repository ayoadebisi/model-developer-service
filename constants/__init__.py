import os

STAGE = os.environ['STAGE']

LOCAL_TRAINING_DATA = '/Users/hadebisi/Documents/PersonalWorkspace/Data/europe_training_data_adj.json'

TRAINING_DATA_COLUMNS = ['home_team', 'away_team', 'home_goal', 'away_goal']

ACTIVE_MODELS = {
    'poisson': None
}
