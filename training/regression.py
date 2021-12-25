from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


from constants import BEST_RATED_MODELS, TEST_SIZE, SHUFFLE, SEED, ACTIVE_MODELS, DROPPABLE_COLUMNS
from helper.model_builder_helper import print_performance
from training.classification import get_classification_features


def train_league_regression(data):
    classification_features = data[get_classification_features()]
    regression_features = data[get_regression_features()]
    labels = data[['home_goal', 'away_goal']]
    results = build_model(regression_features, classification_features, labels)
    if validate_results(results):
        print('Results contains all zeros for home or away, rebuilding regression model.')
        train_league_regression(data)


def get_regression_features():
    return ['goal_difference', 'clean_sheet', 'scoring_streak', 'head_to_head_cs', 'head_to_head_goal',
            'head_to_head_goal_avg', 'head_to_head_scoring']


def build_model(features, classification_features, labels):
    classification_outputs = get_classification_data(classification_features)
    input_data = process_features(features, classification_outputs)

    x_train, x_rem, y_train, y_rem = train_test_split(input_data, labels, test_size=TEST_SIZE,
                                                      shuffle=SHUFFLE, random_state=SEED)

    x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5,
                                                    shuffle=SHUFFLE, random_state=SEED)

    model = model_pipeline(x_train, y_train)
    accuracy = print_performance(model, x_val, x_test, y_val, y_test, False)
    update_best_model(model, accuracy)

    return model.predict(x_test)


def model_pipeline(x, y):
    model = LinearRegression(fit_intercept=True, normalize=True, positive=False)
    model.fit(x, y)
    return model


def process_features(features, classification_features):
    features[['away_win', 'tie', 'home_win']] = classification_features
    return features


def get_classification_data(classification_features):
    return ACTIVE_MODELS['classification'].predict_proba(classification_features)


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
