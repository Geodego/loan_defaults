"""
Tests for data_tools.py
"""
import pandas as pd
import numpy as np
import pytest
from ..data_tools import get_data, select_accounts_with_history, get_df_with_history
from ..data_tools import get_monthly_flows, build_training_template_6m, get_initial_balance, build_training_data


@pytest.fixture()
def mock_accounts():
    """
    Create a mock mock_accounts dataframe
    """
    df = pd.DataFrame({'id': [0, 1], 'balance': [13, 10], 'update_date': ['2021-08-02', '2021-08-02']})
    df['update_date'] = pd.to_datetime(df['update_date'])
    return df


@pytest.fixture()
def mock_transactions():
    """
    Create a mock mock_transactions dataframe
    """
    df = pd.DataFrame({'account_id': [0, 0, 1, 1], 'date': ['2020-11-02', '2021-02-02', '2021-06-02', '2021-07-02'],
                       'amount': [200, -100, 300, -200]})
    df['date'] = pd.to_datetime(df['date'])
    return df


@pytest.fixture()
def accounts():
    """
    Accounts dataframe based on real data
    """
    file_path = "data/accounts.csv"
    return get_data(file_path)


@pytest.fixture()
def transactions():
    """
    Transaction dataframe based on real data
    """
    file_path = "data/transactions.csv"
    return get_data(file_path)


@pytest.fixture()
def update_date(accounts):
    """
    Date at which the update on all accounts has been made for the study
    """
    return accounts['update_date'].iloc[0]


@pytest.fixture()
def flows(accounts, transactions, update_date):
    """
    return inflow and outflow
    """
    accounts, transactions = get_df_with_history(accounts, transactions)
    inflow, outflow = get_monthly_flows(transactions, update_date)
    return inflow, outflow


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


def test_select_accounts_with_history(mock_transactions):
    """
    test select_account_with_history on mock mock_transactions.
    """
    expected_accounts = {0}
    actual_accounts = select_accounts_with_history(mock_transactions)
    assert expected_accounts == actual_accounts


class Test_Get_df_With_History:
    """Regroup tests for get_df_with_history"""

    def test_get_df_with_history_mock_data(self, mock_accounts, mock_transactions):
        new_accounts, new_transactions = get_df_with_history(mock_accounts, mock_transactions)
        assert set(new_accounts['id']) == {0}
        assert set(new_transactions['account_id']) == {0}

    def test_get_df_with_history_real_data(self, accounts, transactions):
        new_accounts, new_transactions = get_df_with_history(accounts, transactions)
        assert set(new_accounts['id']) == {0}
        assert set(new_transactions['account_id']) == {0}


def test_get_monthly_flows(update_date, accounts, transactions):
    accounts, transactions = get_df_with_history(accounts, transactions)
    inflow, outflow = get_monthly_flows(transactions, update_date)


def test_get_initial_balance(accounts, flows):
    inflow, outflow = flows
    balances = get_initial_balance(accounts, inflow, outflow)


def test_build_training_template_6m(accounts, flows):
    inflow, outflow = flows
    inflow_6m = inflow.iloc[:, :7]
    outflow_6m = outflow.iloc[:, :7]
    final_balance = accounts.set_index('id')['balance']
    template = build_training_template_6m(final_balance, inflow_6m, outflow_6m)


def test_build_training_data(accounts, flows):
    inflow, outflow = flows
    initial_balances = get_initial_balance(accounts, inflow, outflow)
    df = build_training_data(initial_balances, inflow, outflow)


