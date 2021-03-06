"""
This file is used to define the objective function used by Optuna for fine-tuning hyperparameters.
"""

import numpy as np
import pandas as pd
from optuna import Trial
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.metrics import mean_squared_error


def objective(trial: Trial, train_x: pd.DataFrame, val_x: pd.DataFrame,
              train_y: np.array, val_y: np.array) -> float:
    """
    Optimisation function using Optuna. The objective of our function is to minimize the RMSE
    :param trial: Trial Optuna object
    :param train_x: features used for training
    :param val_x: features used for validation
    :param train_y: target variable for training
    :param val_y: trarget variable for validation
    :return:
    rmse
    """
    param = {
        'metric': 'rmse',
        'random_state': 48,
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 250, 500, 750, 1000, 1250, 2000]),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [2, 3, 5, 7, 10]),
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        #'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100)
    }
    model = LGBMRegressor(**param)
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[early_stopping(100), log_evaluation(100)])
    preds = model.predict(val_x)
    rmse = mean_squared_error(val_y, preds, squared=False)

    return rmse


def get_objective(train_x: pd.DataFrame, val_x: pd.DataFrame, train_y: np.array, val_y: np.array):
    """
    Returns the optimization function as used by optuna given our dataset. The function returned is a function of trial.
    :param train_x: features used for training
    :param val_x: features used for validation
    :param train_y: target variable for training
    :param val_y: trarget variable for validation
    :return:
    optimization function
    """
    return lambda trial: objective(trial, train_x, val_x, train_y, val_y)
