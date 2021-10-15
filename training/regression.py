from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


from constants import BEST_RATED_MODELS, TEST_SIZE, SHUFFLE, SEED, LEARNING_RATE, DECAY_RATE, \
    EPOCHS, BATCH_SIZE, VERBOSE, ACTIVE_MODELS
from training import classification


def train_league_regression(data):
    classification_features = classification.select_features(data)
    labels = data[['home_goal', 'away_goal']]
    results = build_model(data, classification_features, labels)
    if validate_results(results):
        print('Results contains all zeros for home or away, rebuilding regression model.')
        train_league_regression(data)


def build_model(features, classification_features, labels):
    classification_outputs = get_classification_data(classification_features)
    input_data = process_features(features, classification_outputs)

    x_train, x_rem, y_train, y_rem = train_test_split(input_data, labels, test_size=TEST_SIZE,
                                                      shuffle=SHUFFLE, random_state=SEED)

    x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5,
                                                    shuffle=SHUFFLE, random_state=SEED)

    model = Sequential()

    model.add(Dense(units=x_train.shape[1], activation='softplus', input_dim=x_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(units=75, activation='softplus'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation='softplus'))

    optimizer = Adam(lr=LEARNING_RATE, decay=DECAY_RATE)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    update_best_model(model, results[1])

    print('Regression model results for were:', results)

    return model.predict(x_test)


def process_features(features, classification_features):
    input_data = {
        'home_win': classification_features[:, 1],
        'away_win': classification_features[:, 2],
        'tie': classification_features[:, 0],
        'head_to_head_goal': features['head_to_head_goal'],
        'head_to_head_goal_avg': features['head_to_head_goal_avg'],
        'head_to_head_cs': features['head_to_head_cs'],
        'clean_sheet': features['clean_sheet']
    }

    return DataFrame(input_data)


def get_classification_data(classification_features):
    return ACTIVE_MODELS['classification'].predict(classification_features)


def update_best_model(model, accuracy):
    if accuracy > BEST_RATED_MODELS['regression']:
        print('Better regression model has been trained for and is being uploaded.')
        BEST_RATED_MODELS['regression'] = accuracy
        ACTIVE_MODELS['regression'] = model


def validate_results(results):
    home_win = 0

    for row in results:
        count = 1 if (row[0] > row[1]) else 0
        home_win = home_win + count

    if (results[:, 0] == 0).all() or (results[:, 1] == 0).all():
        return True

    print(f'Home win ratio for this batch of results {(home_win / len(results))}')
    print(f'Score min {results.min()} and max {results.max()}')

    return (home_win / len(results)) > 0.7
