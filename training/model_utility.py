from pandas import DataFrame
from sklearn import preprocessing


def get_features(data, columns):
    features = data[columns]
    min_max_scaler = preprocessing.MinMaxScaler()
    features[['h_off_elo', 'h_def_elo', 'h_pef_elo', 'a_off_elo', 'a_def_elo', 'a_pef_elo']] = \
        DataFrame(min_max_scaler.fit_transform(features.iloc[:, :6].values))
    return features


def get_labels(data, columns):
    return data[columns]
