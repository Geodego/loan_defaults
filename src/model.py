"""
Library of methods for model manipulation.
"""

from .tools import get_absolute_path
from .data_tools import process_data
from joblib import dump, load


def save_model(model, file_name):
    """

    :param model:
    :param file_name:  ex 'lgb'
    :return:
    """
    file_path = 'models/' + file_name + '.joblib'
    dump(model, get_absolute_path(file_path))


def load_model(file_name):
    file_path = 'models/' + file_name + '.joblib'
    model = load(get_absolute_path(file_path))
    return model


def predict(account, transaction):
    """make prediction given input of the API"""
    template = process_data(account, transaction)
    model = load_model('lgb')
    prediction = model.predict(template)[0]
    return prediction


