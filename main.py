from datetime import date
from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, validator
from src import model


class Account(BaseModel):
    update_date: date
    balance: float


class Transaction(BaseModel):
    amount: float
    date: date


class RequestPredict(BaseModel):
    account: Account
    transactions: List[Transaction]

    @validator("transactions")
    def validate_transaction_history(cls, v, *, values):
        # validate that 
        # - the transaction list passed has at least 6 months history
        # - no transaction is posterior to the account's update date
        if len(v) < 1:
            raise ValueError("Must have at least one transaction")

        update_t = values["account"].update_date

        oldest_t = v[0].date
        newest_t = v[0].date
        for t in v[1:]:
            if t.date < oldest_t:
                oldest_t = t.date
            if t.date > newest_t:
                newest_t = t.date

        assert (
                       update_t - newest_t
               ).days >= 0, "Update Date Inconsistent With Transaction Dates"
        assert (update_t - oldest_t).days > 183, "Not Enough Transaction History"

        return v


class ResponsePredict(BaseModel):
    predicted_amount: float


def predict(
        transactions: List[Transaction], account: Account
) -> float:
    account_df = pd.DataFrame({'balance': [account.balance], 'update_date': [account.update_date]})
    account_df['update_date'] = pd.to_datetime(account_df['update_date'])
    transaction_df = pd.DataFrame(map(dict, transactions))
    transaction_df['date'] = pd.to_datetime(transaction_df['date'])
    transaction_df['account_id'] = 0  # the code assume there is an account id
    account_df['id'] = 0
    prediction = model.predict(account_df, transaction_df)
    # raise NotImplementedError()
    return prediction


app = FastAPI()


@app.post("/predict")
async def root(predict_body: RequestPredict):
    transactions = predict_body.transactions
    account = predict_body.account

    # Call your prediction function/code here
    ####################################################
    predicted_amount = predict(transactions, account)

    # Return predicted amount
    return {"predicted_amount": predicted_amount}
