from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from scipy.stats import uniform


from constants import SEED, TEST_SIZE, SHUFFLE, BEST_RATED_MODELS, ACTIVE_MODELS
from helper.model_builder_helper import print_performance


def train_league_classification(data):
    features = select_features(data)
    labels = data['outcome']
    build_model(features, labels)


def build_model(features, labels):
    training_label = LabelEncoder().fit(labels).transform(labels)

    x_train, x_rem, y_train, y_rem = train_test_split(features, training_label, test_size=TEST_SIZE,
                                                      shuffle=SHUFFLE, random_state=SEED)

    x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5,
                                                    shuffle=SHUFFLE, random_state=SEED)

    model = model_pipeline(x_train, y_train)
    accuracy = print_performance(model, x_val, x_test, y_val, y_test, True)
    update_best_model(model, accuracy)


def select_features(dataframe):
    return dataframe[['position', 'performance', 'head_to_head_form', 'unbeaten', 'form', 'winning',
                      'head_to_head_unbeaten', 'head_to_head_winning', 'head_to_head_wins']]


def model_pipeline(x, y):
    model = Pipeline([
        ('sgdclassifier', SGDClassifier(loss='log', alpha=0.038, l1_ratio=0.391, penalty='l1'))
    ])
    '''
    distribution = {
        'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],
        'sgdclassifier__alpha': uniform(0, 0.1),
        'sgdclassifier__l1_ratio': uniform(0, 1)
    }
    model = RandomizedSearchCV(model, param_distributions=distribution)
    print(f'Model best params={model.best_params_}')
    '''
    model.fit(x, y)
    return model


def update_best_model(model, accuracy):
    if accuracy > BEST_RATED_MODELS['classification']:
        print('Better classification model has been trained for and is being uploaded.')
        BEST_RATED_MODELS['classification'] = accuracy
        ACTIVE_MODELS['classification'] = model
