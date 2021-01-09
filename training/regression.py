from pandas import DataFrame
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam


from constants import REGRESSION_TRAINING_FEATURE_COLUMNS, BEST_RATED_MODELS, TEST_SIZE, SHUFFLE, SEED, \
    LEARNING_RATE, DECAY_RATE, EPOCHS, BATCH_SIZE, VERBOSE, ACTIVE_MODELS, CLASSIFICATION_TRAINING_FEATURE_COLUMNS
from training.model_utility import get_features, get_labels


async def train_league_regression(data, country):
    features = get_features(data, REGRESSION_TRAINING_FEATURE_COLUMNS, False)
    classification_features = get_features(data, CLASSIFICATION_TRAINING_FEATURE_COLUMNS, True)
    labels = get_labels(data, ['home_goal', 'away_goal'])
    build_model(features, classification_features, labels, country)


def build_model(features, classification_features, labels, country):
    input_data = process_features(features, classification_features, country)

    x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size=TEST_SIZE,
                                                        shuffle=SHUFFLE, random_state=SEED)

    model = Sequential()

    model.add(Dense(units=x_train.shape[1], activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=2, activation='relu'))

    optimizer = adam(lr=LEARNING_RATE, decay=DECAY_RATE)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    update_best_model(model, results[1], country)

    print('Regression model results for ' + country.capitalize() + ' were:', results)


def process_features(features, classification_features, country):
    classification_outputs = get_classification_data(classification_features, country)

    input_data = {
        'goal_difference': features['home_gd'] - features['away_gd'],
        'scoring_streak': features['h_scoring'] - features['a_scoring'],
        'clean_sheet_streak': features['h_clean_sheet'] - features['a_clean_sheet'],
        'over': features['over'],
        'under': features['under'],
        'home_win': classification_outputs[:, 1],
        'away_win': classification_outputs[:, 2],
        'tie': classification_outputs[:, 0]
    }

    return DataFrame(input_data)


def get_classification_data(classification_features, country):
    classification_request = {
        'offense': classification_features['h_off_elo'] - classification_features['a_off_elo'],
        'defense': classification_features['h_def_elo'] - classification_features['a_def_elo'],
        'performance': classification_features['h_pef_elo'] - classification_features['a_pef_elo'],
        'position': classification_features['home_pos'] - classification_features['away_pos'],
        'form': classification_features['h_form'] - classification_features['a_form'],
        'winning_streak': classification_features['h_winning'] - classification_features['a_winning'],
        'unbeaten_streak': classification_features['h_unbeaten'] - classification_features['a_unbeaten'],
        'home_form': classification_features['h_home'] - classification_features['a_away'],
        'away_form': classification_features['h_away'] - classification_features['a_home'],
        'home_odds': classification_features['home_odds'],
        'away_odds': classification_features['away_odds'],
        'draw_odds': classification_features['draw_odds'],
        'handicap': classification_features['handicap']
    }

    classification_request = DataFrame.from_dict(classification_request)

    return ACTIVE_MODELS['classification'][country.lower()].predict(classification_request)


def update_best_model(model, accuracy, country):
    if accuracy > BEST_RATED_MODELS['regression'][country.lower()]:
        print('Better regression model has been trained for ' + country.capitalize() + ' and is being uploaded.')
        model._make_predict_function()
        BEST_RATED_MODELS['regression'][country.lower()] = accuracy
        ACTIVE_MODELS['regression'][country.lower()] = model
