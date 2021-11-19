import numpy as np

from sklearn.metrics import f1_score, mean_squared_error, r2_score


def print_performance(model, x_val, x_test, y_val, y_test, classification):
    val_predicted = model.predict(x_val)
    test_predicted = model.predict(x_test)

    if classification:
        validation_result = np.mean(val_predicted == y_val)
        test_result = np.mean(test_predicted == y_test)

        val_micro, val_macro = get_f1(y_val, val_predicted)
        test_micro, test_macro = get_f1(y_test, test_predicted)

        print(f'Classification results for validation set: Accuracy={validation_result}, '
              f'Micro={val_micro} & Macro={val_macro}')
        print(f'Classification results for test set: Accuracy={test_result}, '
              f'Micro={test_micro} & Macro={test_macro}')

        return test_result
    else:
        validation_r2, validation_mse = get_regression_metrics(y_val, val_predicted)
        test_r2, test_mse = get_regression_metrics(y_test, test_predicted)

        print(f'Regression results for validation set: R2={validation_r2} & MSE={validation_mse}.')
        print(f'Regression results for test set: R2={test_r2} & MSE={test_mse}.')

        return test_r2


def get_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro'), f1_score(y_true, y_pred, average='macro')


def get_regression_metrics(y_true, y_pred):
    return r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred)
