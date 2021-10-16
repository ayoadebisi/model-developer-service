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

LOCAL_TRAINING_DATA = '/Users/hadebisi/Documents/PersonalWorkspace/Data/europe_training_data.json'

TRAINING_DATA_COLUMNS = ['home_team', 'away_team', 'home_goal', 'away_goal', 'outcome', 'h_off_elo', 'h_def_elo',
                         'h_pef_elo', 'a_off_elo', 'a_def_elo', 'a_pef_elo', 'home_pos', 'home_gd', 'home_pts',
                         'away_pos', 'away_gd', 'away_pts', 'h_form', 'h_winning', 'h_unbeaten', 'h_home', 'h_away',
                         'h_clean_sheet', 'h_scoring', 'a_form', 'a_winning', 'a_unbeaten', 'a_home', 'a_away',
                         'a_clean_sheet', 'a_scoring', 'head_to_head_clean_sheet_1', 'head_to_head_form_1',
                         'head_to_head_goal_1', 'head_to_head_goal_avg_1', 'head_to_head_scoring_1',
                         'head_to_head_unbeaten_1', 'head_to_head_winning_1', 'head_to_head_wins_1',
                         'head_to_head_clean_sheet_2', 'head_to_head_form_2',
                         'head_to_head_goal_2', 'head_to_head_goal_avg_2', 'head_to_head_scoring_2',
                         'head_to_head_unbeaten_2', 'head_to_head_winning_2', 'head_to_head_wins_2',
                         'home_odds', 'draw_odds', 'away_odds', 'over', 'under',
                         'handicap']

NORMALIZED_COLUMNS = ['home_pos', 'away_pos', 'h_winning', 'h_unbeaten', 'h_home', 'h_away', 'h_clean_sheet',
                      'h_scoring', 'a_winning', 'a_unbeaten', 'a_home', 'a_away', 'a_clean_sheet', 'a_scoring',
                      'head_to_head_clean_sheet_1']

STANDARDIZED_COLUMNS = ['h_off_elo', 'h_def_elo', 'h_pef_elo', 'a_off_elo', 'a_def_elo', 'a_pef_elo', 'home_gd',
                        'home_pts', 'away_gd', 'away_pts', 'h_form', 'a_form', 'head_to_head_form_1']

BEST_RATED_MODELS = {
    'classification': 0.0,
    'regression': 0.0
}

ACTIVE_MODELS = {
    'classification': None,
    'regression': None,
    'poisson': None
}

NORMALIZATION_KEYS = {
    'PerformanceElo': 'performance',
    'OffensiveRating': '',
    'WinningStreak': 'winning',
    'Form': 'form',
    'UnbeatenStreak': 'unbeaten',
    'DefensiveElo': 'defense',
    'CleanSheet': 'clean_sheet',
    'Points': 'points',
    'OffensiveElo': 'offense',
    'AwayForm': 'away_form',
    'ScoringStreak': 'scoring_streak',
    'HomeForm': 'home_form',
    'Position': 'position',
    'DefensiveRating': '',
    'GoalDifference': 'goal_difference',
    'PerformanceRating': '',
    'AwayCleanSheet': '',
    'HeadToHeadGoal': 'head_to_head_goal',
    'HeadToHeadGoalAvg': 'head_to_head_goal_avg',
    'HeadToHeadCS': 'head_to_head_cs',
    'HeadToHeadForm': 'head_to_head_form',
    'HeadToHeadScoring': 'head_to_head_scoring',
    'HeadToHeadUnbeaten': 'head_to_head_unbeaten',
    'HeadToHeadWinning': 'head_to_head_winning',
    'HeadToHeadWins': 'head_to_head_wins',
    'HomeTeam': '',
    'AwayTeam': ''
}
