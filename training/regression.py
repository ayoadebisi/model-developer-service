from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import hashing_trick


from constants import BEST_RATED_MODELS, TEST_SIZE, SHUFFLE, SEED, LEARNING_RATE, DECAY_RATE, \
    EPOCHS, BATCH_SIZE, VERBOSE, ACTIVE_MODELS
from training import classification

NUM_TEAMS = {'Length': 0}


def train_league_regression(data):
    classification_features = classification.select_features(data)
    labels = data[['home_goal', 'away_goal']]
    build_model(data, classification_features, labels)


def build_model(features, classification_features, labels):
    classification_outputs = get_classification_data(classification_features)
    input_data = process_features(features, classification_outputs)

    x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size=TEST_SIZE,
                                                        shuffle=SHUFFLE, random_state=SEED)

    x_train, x_val, y_train, y_val = train_test_split(input_data, labels, test_size=TEST_SIZE,
                                                      shuffle=SHUFFLE, random_state=SEED)

    model = Sequential()

    model.add(Dense(units=x_train.shape[1], activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=2, activation='relu'))

    optimizer = Adam(lr=LEARNING_RATE, decay=DECAY_RATE)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    update_best_model(model, results[1])

    print('Regression model results for were:', results)


def process_features(features, classification_features):
    global NUM_TEAMS
    NUM_TEAMS['Length'] = len(set(features['home_team']))

    input_data = {
        'home_win': classification_features[:, 1],
        'away_win': classification_features[:, 2],
        'tie': classification_features[:, 0],
        'away_clean_sheet': features['a_clean_sheet'],
        'home_team': [hash_team_name(team, NUM_TEAMS['Length']) for team in features['home_team']],
        'away_team': [hash_team_name(team, NUM_TEAMS['Length']) for team in features['away_team']]
    }

    return DataFrame(input_data)


def hash_team_name(team, length):
    return hashing_trick(team, round(length * 1.3), hash_function='md5')[0]


def get_classification_data(classification_features):
    return ACTIVE_MODELS['classification'].predict(classification_features)


def update_best_model(model, accuracy):
    if accuracy > BEST_RATED_MODELS['regression']:
        print('Better regression model has been trained for and is being uploaded.')
        BEST_RATED_MODELS['regression'] = accuracy
        ACTIVE_MODELS['regression'] = model
