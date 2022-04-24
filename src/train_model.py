"""
Training and evaluation of the model.
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from .data_tools import get_training_data
from .hyper_params import get_objective


def train_lgbm(test_size: int = 2):
    """
    Train an LightGBM model on data in the data folder. Data are processed then hyperparameters tuned and finally the
    model is trained using the optimal set of hyperparameters.
    :param test_size: number of months to keep for testing, set by default to the las 2 months. The same number of
    months is used for validation.
    :return:
    """
    training_data = get_training_data(test_size=test_size)
    x_train = training_data['train']['X']
    y_train = training_data['train']['y']
    x_val = training_data['val']['X']
    y_val = training_data['val']['y']

    # Hyperparameters tuning
    study = optuna.create_study(direction='minimize')
    objective = get_objective(x_train, x_val, y_train, y_val)
    study.optimize(objective, n_trials=50)
    params = study.best_params
    params['random_state'] = 48
    params['metric'] = 'rmse'
    print(params)
    model = LGBMRegressor(**params)
    model.fit(x_train, y_train)
    return model


def eval_model(model: LGBMRegressor, x_test: pd.DataFrame = None, y_test: pd.DataFrame = None,
               test_size: int = 2) -> dict:
    """
    Model evaluation on test data. Use x_test and y_test if they are provided, otherwise get them from data folder.
    :param model:
    :param x_test:
    :param y_test:
    :param test_size:
    :return:
    dictionary {'r2': , 'mse': , 'rmse_percent': } 'rmse_percent' gives the rmse as a percentage of the average loss.
    """
    if x_test is None or y_test is None:
        training_data = get_training_data(test_size=test_size)
        x_test = training_data['test']['X']
        y_test = training_data['test']['y']
    y_predict = model.predict(x_test)
    eval = {}
    eval['r2'] = r2_score(y_test, y_predict)
    eval['rmse'] = np.sqrt(mean_squared_error(y_test, y_predict))
    eval['rmse_percent'] = round(eval['rmse'] / np.abs(y_test.mean()) * 100, 1)
    return eval


if __name__ == '__main__':
    model = train_lgbm()
    result = eval_model(model)
    print(result)

