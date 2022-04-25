import pandas as pd
from datetime import date
from ..model import predict


def test_predict():
    """
    Test the predict function with mock data. This test just check that the calculation is done without raising
    exceptions.
    """
    test_account = {"balance": [10000], "update_date": [str(date(2020, 11, 3))]}
    test_transactions = [
        {"date": str(date(2020, i, j)), "amount": -100}
        for i in range(1, 10)
        for j in [5, 17, 26]
    ]

    account_df = pd.DataFrame(test_account)
    account_df['update_date'] = pd.to_datetime(account_df['update_date'])
    transaction_df = pd.DataFrame({'date': [transaction["date"] for transaction in test_transactions],
                                   'amount': [transaction["amount"] for transaction in test_transactions]})
    transaction_df['date'] = pd.to_datetime(transaction_df['date'])
    transaction_df['account_id'] = 0  # the code assume there is an account id
    account_df['id'] = 0
    _ = predict(account_df, transaction_df)
