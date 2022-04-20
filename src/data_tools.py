"""
Library of functions used to manipulate data
"""

import pandas as pd
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
    Select account ids having sufficient history of transactions
    :param accounts: dataframe with accounts information
    :param transactions: dataframe with history of transactions for different accounts
    :param histo: minimum history required in days
    :return: set of selected accounts id
    """
    # We get the dates of the 1st transaction and the last transaction for each account
    history_df = transactions.groupby('account_id')['date'].agg(["min", "max"])
    #todo: merge history_df with accounts
    history = history_df['max'] - history_df['min']
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
    selected_accounts = select_accounts_with_history(transactions, histo)
    new_accounts = accounts[accounts['id'].isin(selected_accounts)].copy()
    new_transactions = transactions[transactions['account_id'].isin(selected_accounts)].copy()
    return new_accounts, new_transactions

