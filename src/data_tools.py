"""
Library of functions used to manipulate data
"""

import pandas as pd
from pandas.tseries.offsets import Day
from typing import Tuple
from .tools import get_absolute_path


def get_data(local_file_path: str) -> pd.DataFrame:
    """
    Return dataframe corresponding to csv file in local_file_path
    :param local_file_path: local path within the project eg: 'data/accounts.csv'
    :return:
    pd.DataFrame
    """
    df = pd.read_csv(get_absolute_path(local_file_path))

    # we set the 'date' or the 'update_date' columns as a datetime object
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'update_date' in df.columns:
        df['update_date'] = pd.to_datetime(df['update_date'])

    return df


def select_accounts_with_history(accounts: pd.DataFrame, transactions: pd.DataFrame, histo: int = 180
                                 ) -> set:
    """
    Select account ids having sufficient history of transactions. The available transaction history on each account
    is defined as the time elapsed since the oldest transaction recorded on this account and the update date of the
    account, not the date of the latest transaction on the account.
    :param accounts: dataframe with accounts information
    :param transactions: dataframe with history of transactions for different accounts
    :param histo: minimum history required in days
    :return: set of selected accounts id
    """
    # We get the dates of the 1st transaction and the last transaction for each account
    history_df = transactions.groupby('account_id')['date'].agg(["min", "max"])
    merged = pd.merge(history_df, accounts, left_index=True, right_on='id')
    history = merged['update_date'] - merged['min']
    trigger = pd.Timedelta(histo, "d")
    selected_accounts = set(history[history > trigger].index)
    return selected_accounts


def get_df_with_history(accounts: pd.DataFrame, transactions: pd.DataFrame, histo: int = 180
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict accounts and transactions dataframes to selected account with sufficient history.
    The history for a given account is calculated from the first transaction found in transactions
    to the update_date for that account in accounts.
    :param accounts: dataframe with accounts information
    :param transactions: dataframe with history of transactions for different accounts
    :param histo: minimum history required in days
    :return: tuple of accounts and transactions dataframes for selected accounts.
    """
    selected_accounts = select_accounts_with_history(accounts, transactions, histo)
    new_accounts = accounts[accounts['id'].isin(selected_accounts)].copy()
    new_transactions = transactions[transactions['account_id'].isin(selected_accounts)].copy()
    return new_accounts, new_transactions


def get_monthly_flows(transactions: pd.DataFrame, update_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get outflows and inflows over 30 days periods
    :param transactions:
    :param update_date:
    :return:
    pd.Dataframes inflow and outflow. Both have account id as index and a Timestamp corresponding as the last day of
    the 30 days period as column names.
    """
    # we first split inflows and outflows
    df = transactions.copy()
    df['inflow'] = df['amount'].where(df['amount'] > 0, 0)
    df['outflow'] = df['amount'].where(df['amount'] < 0, 0)
    # we want daily balances of inflows and outflows for each account. We need to handle the case where an account has
    # several transactions of the same nature (inflow or out flow) on the same day.
    df = df.groupby(['account_id', 'date'])[['inflow', 'outflow']].sum()

    # min_date is the date of the oldest transaction within our data
    min_date = min(transactions['date'])

    # We want to slice our data in 30 days bucket, with the last bucket ending on the update date. For that purpose
    # we need to calculate the start_date which will allow us to build time buckets ending on update_date.
    # we calculate the number of 30 days periods between min_date and update_date rounded down.
    number_periods = round((update_date - min_date).days / 30)
    start_date = update_date - Day(30 * number_periods)

    # most of the time start_date will fall after min_date (because of rounding). In that case we need to add another
    # 30 day bucket so that we are sure to include all the data.
    if start_date > min_date:
        start_date -= Day(30)

    # we build from df, outflow with sum of outflows for each account on every 30 days period, and inflow with sum of
    # inflows for each account on every 30 days period.
    outflow = df['outflow'].unstack()
    outflow[start_date] = 0
    outflow = outflow.resample('30D', axis=1, label='right').sum()

    inflow = df['inflow'].unstack()
    inflow[start_date] = 0
    inflow = inflow.resample('30D', axis=1, label='right').sum()

    return inflow, outflow


def get_initial_balance(accounts: pd.DataFrame, inflow: pd.DataFrame, outflow: pd.DataFrame) -> pd.Series:
    """
    Get initial account balances
    :param accounts:
    :param inflow:
    :param outflow:
    :return:
    pd.Series with accounts as index and initial balances as values.
    """
    final_balance = accounts.set_index('id')['balance']
    flows = pd.DataFrame({'inflow': inflow.sum(axis=1), 'outflow': outflow.sum(axis=1)})
    flows = flows.merge(final_balance, left_index=True, right_index=True)
    initial_balance = flows['balance'] - flows['outflow'] - flows['inflow']
    return initial_balance


def build_training_template_6m(initial_balance, inflow, outflow) -> pd.DataFrame:
    """
    Build dataframe used for training given initial balance and 7month inflow and outflow. The columns names are
    standardized as 'kM inflow' or 'kM outflow' indicating the kth month of data. The column 'initial_balance', gives
    the balance of the accounts before the first period considered. The last column, 'true_outgoing' will
    be used to supervise the predictions made with the 6 previous months. This will allow us to build 6 month
    history data across different periods and train our model on the resulting data.
    :param initial_balance:
    :param inflow:
    :param outflow:
    :return:
    """
    # we only keep accounts that have a transaction during the period considered
    active_accounts = (inflow != 0).any(axis=1) | (outflow != 0).any(axis=1)
    active_inflow = inflow.copy().loc[active_accounts]
    active_outflow = outflow.copy().loc[active_accounts]
    active_inflow = active_inflow.iloc[:, :6]  # we don't need the 7 month of inflow

    inflow_cols = {col: f'{i + 1}M inflow' for i, col in enumerate(active_inflow.columns)}
    outflow_cols = {col: f'{i + 1}M outflow' for i, col in enumerate(active_outflow.columns)}
    outflow_cols[active_outflow.columns[-1]] = 'true_outgoing'
    active_inflow.rename(columns=inflow_cols, inplace=True)
    active_outflow.rename(columns=outflow_cols, inplace=True)
    template = active_inflow.merge(active_outflow, left_index=True, right_index=True)
    initial_balance.name = 'initial_balance'
    template = template.merge(initial_balance, left_index=True, right_index=True)
    return template


def build_training_data(initial_balance, inflow, outflow):
    """organise the training data rolling over time"""
    n_months = inflow.shape[1]
    df_list = []
    for k in range(n_months-6):
        inflow_7m = inflow.iloc[:, k:k+7]
        outflow_7m =  outflow.iloc[:, k:k+7]
        template = build_training_template_6m(initial_balance, inflow_7m, outflow_7m)
        # add template to our list of training data
        df_list.append(template)
        # update what will be the new initial balances for the next turn. We need to add the inflow and the outflow
        # from the first month of the 7 month period considered
        df_balance = pd.DataFrame({'initial_balance': initial_balance, 'inflow': inflow_7m.iloc[:, 0],
                                   'outflow': outflow_7m.iloc[:, 0]})
        initial_balance = df_balance.sum(axis=1)
    training_data = pd.concat(df_list, axis=0, ignore_index=True)
    return training_data










