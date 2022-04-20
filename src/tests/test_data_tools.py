"""
Tests for data_tools.py
"""
import pandas as pd
import numpy as np
import pytest
from ..data_tools import get_data, select_accounts_with_history, get_df_with_history


@pytest.fixture()
def accounts():
    """
    Create a mock accounts dataframe
    """
    df = pd.DataFrame({'id': [0, 1], 'balance': [13, 10], 'update_date': ['2021-08-02', '2021-07-01']})
    df['update_date'] = pd.to_datetime(df['update_date'])
    return df


@pytest.fixture()
def transactions():
    """
    Create a mock transactions dataframe
    """
    df = pd.DataFrame({'account_id': [0, 0, 1, 1], 'date': ['2020-11-02', '2021-02-02', '2021-06-02', '2021-07-02'],
                       'amount': [200, -100, 300, -200]})
    df['date'] = pd.to_datetime(df['date'])
    return df


@pytest.mark.parametrize('raw_path', ["data/accounts.csv", "data/transactions.csv"])
def test_get_data(self, raw_path):
    """
    Check we get some data from the csv files and if the date columns are properly formated.
    """
    df = get_data(raw_path)
    assert not df.empty
    if 'date' in df.columns:
        assert df['date'].dtype.type == np.datetime64
    if 'update_date' in df.columns:
        assert df['update_date'].dtype.type == np.datetime64


def test_select_accounts_with_history(transactions):
    """
    test select_account_with_history on mock transactions.
    """
    expected_accounts = {0}
    actual_accounts = select_accounts_with_history(transactions)
    assert expected_accounts == actual_accounts


def test_get_df_with_history(accounts, transactions):
    new_accounts, new_transactions = get_df_with_history(accounts, transactions)
    assert set(new_accounts['id']) == {0}
    assert set(new_transactions['account_id']) == {0}

