"""
Tests for data_tools.py
"""

import pandas as pd
import numpy as np
import pytest
from ..data_tools import get_data, select_accounts_with_history, get_df_with_history
from ..data_tools import get_monthly_flows, build_template_6m, get_initial_balance, build_training_data


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
def test_get_data(raw_path):
    """
    Check we get some data from the csv files and if the date columns are properly formated.
    """
    df = get_data(raw_path)
    assert not df.empty
    if 'date' in df.columns:
        assert df['date'].dtype.type == np.datetime64
    if 'update_date' in df.columns:
        assert df['update_date'].dtype.type == np.datetime64


def test_select_accounts_with_history(mock_accounts, mock_transactions):
    """
    test select_account_with_history on mock mock_transactions.
    """
    expected_accounts = {0}
    actual_accounts = select_accounts_with_history(mock_accounts, mock_transactions)
    assert expected_accounts == actual_accounts


class Test_Get_df_With_History:
    """Regroup tests for get_df_with_history"""

    def test_get_df_with_history_mock_data(self, mock_accounts, mock_transactions):
        new_accounts, new_transactions = get_df_with_history(mock_accounts, mock_transactions)
        assert set(new_accounts['id']) == {0}
        assert set(new_transactions['account_id']) == {0}

    def test_get_df_with_history_real_data(self, accounts, transactions):
        histo = 180
        new_accounts, new_transactions = get_df_with_history(accounts, transactions, histo=histo)
        # sample 10 accounts and check they have enough history
        sampled = new_accounts.sample(10)['id'].values
        trigger = pd.Timedelta(histo, "d")
        update_date = accounts['update_date'].iloc[0]
        for id in sampled:
            id_transactions = new_transactions[new_transactions['account_id'] == id]
            first_date = id_transactions['date'].min()
            assert (update_date - first_date).days > histo


def test_get_monthly_flows(update_date, accounts, transactions):
    accounts, transactions = get_df_with_history(accounts, transactions)
    inflow, outflow = get_monthly_flows(transactions, update_date)
    assert len(inflow) == len(outflow)
    # sample 10 accounts and check that the sum of inflows and outflows equals the sum of transactions
    sampled = accounts.sample(10)['id'].values
    for id in sampled:
        expected_flow = transactions[transactions['account_id'] == id]['amount'].sum()
        actual_flow = (inflow.loc[id] + outflow.loc[id]).sum()
        assert np.isclose(expected_flow, actual_flow)


def test_get_initial_balance(accounts, flows):
    inflow, outflow = flows
    initial_balances = get_initial_balance(accounts, inflow, outflow)
    final_balances = accounts.loc[inflow.index]['balance']
    estimated_final_balances = initial_balances + (inflow + outflow).sum(axis=1)
    assert np.allclose(final_balances, estimated_final_balances)


class TestBuildTrainingTemplate6m:
    """
    Tests for build_training_template_6m
    """

    def test_build_training_template_6m_training_mock(self, mock_accounts, mock_transactions):
        """
        test build_training_template_6m on mock data when it's used for training
        """
        update_date = mock_accounts['update_date'].iloc[0]
        inflow, outflow = get_monthly_flows(mock_transactions, update_date)
        inflow_6m = inflow.iloc[:, :7]
        outflow_6m = outflow.iloc[:, :7]
        final_balance = mock_accounts.set_index('id')['balance']
        initial_balance = final_balance - (inflow + outflow).sum(axis=1)
        template = build_template_6m(initial_balance, inflow_6m, outflow_6m, inference=False)
        assert (template['initial_balance'] == initial_balance[template.index]).all()
        assert (template['true_outgoing'] == outflow_6m.loc[template.index].iloc[:, -1]).all()

    def test_build_training_template_6m_training_real(self, accounts, flows):
        """
        test build_training_template_6m on real data when it's used for training
        """
        inflow, outflow = flows
        inflow_6m = inflow.iloc[:, :7]
        outflow_6m = outflow.iloc[:, :7]
        final_balance = accounts.set_index('id')['balance']
        initial_balance = final_balance - (inflow + outflow).sum(axis=1)
        template = build_template_6m(initial_balance, inflow_6m, outflow_6m, inference=False)
        assert (template['initial_balance'] == initial_balance[template.index]).all()
        assert (template['true_outgoing'] == outflow_6m.loc[template.index].iloc[:, -1]).all()

    def test_build_training_template_6m_inference_real(self, accounts, flows):
        """
        test build_training_template_6m on real data when it's used for inference
        """
        inflow, outflow = flows
        inflow_6m = inflow.iloc[:, :7]
        outflow_6m = outflow.iloc[:, :7]
        final_balance = accounts.set_index('id')['balance']
        initial_balance = final_balance - (inflow + outflow).sum(axis=1)
        template = build_template_6m(initial_balance, inflow_6m, outflow_6m, inference=True)
        assert (template['initial_balance'] == initial_balance[template.index]).all()
        assert 'true_outgoing' not in template.columns


def build_training_data_checks(final_balance, initial_balances, inflow, outflow, x_train, x_test, y_test):
    """
    function used by tests for build_training_data
    """
    # we check that we can calculate properly the final balances for one account
    assert np.allclose(final_balance, initial_balances + (inflow + outflow).sum(axis=1))
    # final balance after the 6 first months
    x_train_final_balance = x_train.iloc[:2].sum(axis=1).values
    # net flow from month 7 to 11
    flow_7_to_11 = x_test.iloc[2:, 1:-1].drop(['1M outflow'], axis=1).sum(axis=1).values
    # last month flow (12th month flow)
    last_month_flow = y_test[2:] + inflow.iloc[:, -1].values
    # x_train gives the transactions of the first 6 months + initial balances
    calculated_balances = x_train_final_balance + flow_7_to_11 + last_month_flow

    # final balance after the 6 first months + net flow from month 7 to 11 + last month flow should be equal to
    # final balance
    assert np.allclose(final_balance, calculated_balances)


@pytest.fixture()
def data_split(accounts, flows):
    """
    Get data split for the tests on build_training_data using real data.To make things simple for this test we keep
    the last 12 months of data and keep only 2 accounts
    """
    ids = [4, 7]
    inflow, outflow = flows
    inflow, outflow = inflow.loc[ids].iloc[:, -12:], outflow.loc[ids].iloc[:, -12:]
    initial_balances = get_initial_balance(accounts, inflow, outflow)
    training_data = build_training_data(initial_balances, inflow, outflow)
    x_train, x_val, x_test = training_data['train']['X'], training_data['val']['X'], training_data['test']['X']
    y_test = training_data['test']['y']
    return x_train, x_val, x_test, y_test, inflow, outflow, initial_balances


class TestBuildTrainingData:
    """
    Tests for build_training_data
    """

    def test_build_training_data_repartition(self, data_split):
        """
        Test the repartition between training validation and testing and the shape of features
        """
        x_train, x_val, x_test, _, _, _, _ = data_split
        assert len(x_train) == 4
        assert len(x_val) == 4
        assert len(x_test) == 4

    def test_build_training_data_mock(self, accounts, flows):
        """
        Check that all the steps performed in build_training_data are performed properly. For that purpose we
        reconstitute the final balances from data in x_train and x_test. This version of the test uses mock data.
        """
        # build the mock data
        inflow, outflow = flows
        inflow, outflow = inflow.iloc[:2, -12:], outflow.iloc[:2, -12:]
        cash_pos = np.arange(1, 13)
        cash_neg = - np.arange(1, 13)
        cash_neg[11] -= 1
        inflow.iloc[:] = cash_pos
        outflow.iloc[:] = cash_neg
        final_balance = pd.Series([99, 99])
        mock_accounts = accounts.copy()[accounts['id'].isin(inflow.index)]
        mock_accounts['balance'] = final_balance
        initial_balances = get_initial_balance(mock_accounts, inflow, outflow)  # should be 100
        assert np.allclose(initial_balances, np.array([100, 100]))

        training_data = build_training_data(initial_balances, inflow, outflow)
        x_train, x_val, x_test = training_data['train']['X'], training_data['val']['X'], training_data['test']['X']
        y_test = training_data['test']['y']

        # perform all the necessary checks
        build_training_data_checks(final_balance, initial_balances, inflow, outflow, x_train, x_test, y_test)

    def test_build_training_data_real(self, data_split, accounts):
        """
        Check that all the steps performed in build_training_data are performed properly. For that purpose we
        reconstitute the final balances from data in x_train and x_test. This version of th test uses real data.
        """
        x_train, x_val, x_test, y_test, inflow, outflow, initial_balances = data_split
        # we check that we can calculate properly the final balances for one account
        final_balance = accounts['balance'].loc[initial_balances.index]

        # perform all the necessary checks
        build_training_data_checks(final_balance, initial_balances, inflow, outflow, x_train, x_test, y_test)
