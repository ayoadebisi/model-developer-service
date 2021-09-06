from pandas import DataFrame
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


from constants import SEED, TEST_SIZE, SHUFFLE, BATCH_SIZE, EPOCHS, VERBOSE, \
    LEARNING_RATE, DECAY_RATE, BEST_RATED_MODELS, ACTIVE_MODELS


def train_league_classification(data):
    features = select_features(data)
    labels = data['outcome']
    build_model(features, labels)


def build_model(features, labels):
    training_label = to_categorical(labels, num_classes=3)

    x_train, x_test, y_train, y_test = train_test_split(features, training_label, test_size=TEST_SIZE,
                                                        shuffle=SHUFFLE, random_state=SEED)

    x_train, x_val, y_train, y_val = train_test_split(features, training_label, test_size=TEST_SIZE,
                                                      shuffle=SHUFFLE, random_state=SEED)

    model = Sequential()

    model.add(Dense(units=x_train.shape[1], activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    optimizer = Adam(lr=LEARNING_RATE, decay=DECAY_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    update_best_model(model, results[1])

    print('Classification model results for were:', results)


def select_features(dataframe):
    data = {
        'performance': dataframe['h_pef_elo'] - dataframe['a_pef_elo'],
        'position': dataframe['home_pos'] - dataframe['away_pos'],
        'form': dataframe['h_form'] - dataframe['a_form'],
        'winning': dataframe['h_winning'] - dataframe['a_winning'],
        'unbeaten': dataframe['h_unbeaten'] - dataframe['a_unbeaten'],
        'clean_sheet': dataframe['h_clean_sheet'] - dataframe['a_clean_sheet'],
        'head_to_head_cs': dataframe['head_to_head_clean_sheet_1'] - dataframe['head_to_head_clean_sheet_2'],
        'head_to_head_form': dataframe['head_to_head_form_1'] - dataframe['head_to_head_form_2'],
        'head_to_head_goal': dataframe['head_to_head_goal_1'] - dataframe['head_to_head_goal_2'],
        'head_to_head_goal_avg': dataframe['head_to_head_goal_avg_1'] - dataframe['head_to_head_goal_avg_2'],
        'head_to_head_scoring': dataframe['head_to_head_scoring_1'] - dataframe['head_to_head_scoring_2'],
        'head_to_head_unbeaten': dataframe['head_to_head_unbeaten_1'] - dataframe['head_to_head_unbeaten_2'],
        'head_to_head_winning': dataframe['head_to_head_winning_1'] - dataframe['head_to_head_winning_2'],
        'head_to_head_wins': dataframe['head_to_head_wins_1'] - dataframe['head_to_head_wins_2'],
        'away': dataframe['a_away']
    }

    return DataFrame(data)


def update_best_model(model, accuracy):
    if accuracy > BEST_RATED_MODELS['classification']:
        print('Better classification model has been trained for and is being uploaded.')
        BEST_RATED_MODELS['classification'] = accuracy
        ACTIVE_MODELS['classification'] = model
