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

    x_train, x_rem, y_train, y_rem = train_test_split(features, training_label, test_size=TEST_SIZE,
                                                      shuffle=SHUFFLE, random_state=SEED)

    x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5,
                                                    shuffle=SHUFFLE, random_state=SEED)

    model = Sequential()

    model.add(Dense(units=x_train.shape[1], activation='softmax', input_dim=x_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(units=3, activation='softmax'))

    optimizer = Adam(lr=LEARNING_RATE, decay=DECAY_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    update_best_model(model, results[1])

    print('Classification model results for were:', results)


def select_features(dataframe):
    return dataframe[['position', 'performance', 'head_to_head_form', 'unbeaten', 'form', 'winning',
                      'head_to_head_unbeaten', 'head_to_head_winning', 'head_to_head_wins']]


def update_best_model(model, accuracy):
    if accuracy > BEST_RATED_MODELS['classification']:
        print('Better classification model has been trained for and is being uploaded.')
        BEST_RATED_MODELS['classification'] = accuracy
        ACTIVE_MODELS['classification'] = model
