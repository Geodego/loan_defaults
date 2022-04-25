"""
Library of methods for model manipulation.
"""
import pandas as pd

from .tools import get_absolute_path
from .data_tools import process_data
from joblib import dump, load
from lightgbm import LGBMRegressor


def save_model(model: LGBMRegressor, file_name: str):
    """
    Save the model as a joblib file in folder models. If file_name='lgb', save as 'lgb.joblib'
    :param model:
    :param file_name:  ex 'lgb'
    :return:
    """
    file_path = 'models/' + file_name + '.joblib'
    dump(model, get_absolute_path(file_path))


def load_model(file_name: str) -> LGBMRegressor:
    """
    Load the model saved in folder models with the name file_name
    :param file_name: name of the file where is saved the model (ex: 'lgb')
    """
    file_path = 'models/' + file_name + '.joblib'
    model = load(get_absolute_path(file_path))
    return model


def predict(account: pd.DataFrame, transaction: pd.DataFrame) -> float:
    """
    Make a prediction of the next month outgoing given input of the API
    """
    template = process_data(account, transaction)
    model = load_model('lgb')
    prediction = model.predict(template)[0]
    return prediction
