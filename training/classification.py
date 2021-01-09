from pandas import DataFrame
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam


from constants import CLASSIFICATION_TRAINING_FEATURE_COLUMNS, SEED, TEST_SIZE, SHUFFLE, BATCH_SIZE, EPOCHS, VERBOSE, \
    LEARNING_RATE, DECAY_RATE, BEST_RATED_MODELS, ACTIVE_MODELS
from training.model_utility import get_features, get_labels


async def train_league_classification(data, country):
    features = get_features(data, CLASSIFICATION_TRAINING_FEATURE_COLUMNS)
    labels = get_labels(data, ['outcome'])
    build_model(features, labels, country)


def build_model(features, labels, country):
    input_data = process_features(features)
    training_label = to_categorical(labels, num_classes=3)

    x_train, x_test, y_train, y_test = train_test_split(input_data, training_label, test_size=TEST_SIZE,
                                                        shuffle=SHUFFLE, random_state=SEED)

    model = Sequential()

    model.add(Dense(units=x_train.shape[1], activation='linear', input_dim=x_train.shape[1]))
    model.add(Dense(units=20, activation='linear'))
    model.add(Dropout(0.5))
    model.add(Dense(units=20, activation='linear'))
    model.add(Dropout(0.5))
    model.add(Dense(units=3, activation='softmax'))

    optimizer = adam(lr=LEARNING_RATE, decay=DECAY_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    update_best_model(model, results[1], country)

    print('Classification model results for ' + country.capitalize() + ' were:', results)


def process_features(features):
    input_data = {
        'offense': features['h_off_elo'] - features['a_off_elo'],
        'defense': features['h_def_elo'] - features['a_def_elo'],
        'performance': features['h_pef_elo'] - features['a_pef_elo'],
        'position': features['home_pos'] - features['away_pos'],
        'form': features['h_form'] - features['a_form'],
        'winning_streak': features['h_winning'] - features['a_winning'],
        'unbeaten_streak': features['h_unbeaten'] - features['a_unbeaten'],
        'home_form': features['h_home'] - features['a_away'],
        'away_form': features['h_away'] - features['a_home'],
        'home_odds': features['home_odds'],
        'away_odds': features['away_odds'],
        'draw_odds': features['draw_odds'],
        'handicap': features['handicap']
    }

    return DataFrame(input_data)


def update_best_model(model, accuracy, country):
    if accuracy > BEST_RATED_MODELS['classification'][country.lower()]:
        print('Better classification model has been trained for ' + country.capitalize() + ' and is being uploaded.')
        model._make_predict_function()
        BEST_RATED_MODELS['classification'][country.lower()] = accuracy
        ACTIVE_MODELS['classification'][country.lower()] = model
