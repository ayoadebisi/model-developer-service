from pandas import DataFrame
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam


from constants import REGRESSION_TRAINING_FEATURE_COLUMNS, BEST_RATED_MODELS, TEST_SIZE, SHUFFLE, SEED,\
    LEARNING_RATE, DECAY_RATE, EPOCHS, BATCH_SIZE, VERBOSE, ACTIVE_MODELS
from training.model_utility import get_features, get_labels


async def train_league_regression(data, country):
    features = get_features(data, REGRESSION_TRAINING_FEATURE_COLUMNS)
    labels = get_labels(data, ['home_goal', 'away_goal'])
    build_model(features, labels, country)


def build_model(features, labels, country):
    input_data = process_features(features)

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


def process_features(features):
    input_data = {
        'offense_1': features['h_off_elo'] - features['a_def_elo'],
        'offense_2': features['a_off_elo'] - features['h_def_elo'],
        'performance': features['h_pef_elo'] - features['a_pef_elo'],
        'goal_difference': features['home_gd'] - features['away_gd'],
        'scoring_streak': features['h_scoring'] - features['a_scoring'],
        'clean_sheet_streak': features['h_clean_sheet'] - features['a_clean_sheet'],
        'over': features['over'],
        'under': features['under'],
        'handicap': features['handicap']
    }

    return DataFrame(input_data)


def update_best_model(model, accuracy, country):
    if accuracy > BEST_RATED_MODELS['regression'][country.lower()]:
        print('Better regression model has been trained for ' + country.capitalize() + ' and is being uploaded.')
        model._make_predict_function()
        BEST_RATED_MODELS['regression'][country.lower()] = accuracy
        ACTIVE_MODELS['regression'][country.lower()] = model
