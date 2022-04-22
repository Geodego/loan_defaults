<p align="center"><a href="https://github.com/MansaGroup/kanedama" target="blank"><img src="../.github/assets/logo.png" width="80" alt="Mansa's Logo" /></a></p>
<h1 align="center">Mansa's Kanedama</h1>
<p align="center">Take home test to <b>join us</b> 💜</p>

# The Mission

Your mission, should you choose to accept it, is to demonstrate your ability to deal with time-series data, build a model, and serve it using an API framework. Your task is to use the provided data  (more info at the bottom) to design a model capable of making predictions.

Your model should then be served through the small [FastAPI](https://fastapi.tiangolo.com/) file provided. 

## Before you pick a model

You will deal with bank account transactions data. We suggest you take a look and explore them. Since those are real data, they are noisy and sparse, and there may be duplicated data.

Build a function to check which accounts have more than 180 days of history - you can discard the others for your models and analysis.

_You can assume that any account passed to your service will have at least 6 months of history._

## Predict next month outgoing given the past 6 months of transactions

Set up a prediction function that takes an account, a list of transactions recorded on the account, and ouputs a prediction for the aggregated next month outgoing for that account.

Outgoing is defined as the sum of all transactions with `< 0` amount over a certain time-period. So to get the monthly outgoing, you can sum the negative transactions over monthly periods.

> **Tip 1.** It might make more sense to define a month as a 30 day period rather than the month itself since the snapshots can be taken at any point during the month and not necessarily at the end. 

# Delivery

To achieve your mission, you'll have to deliver:

- A documented `Python` (3.x) code (please specify what `Python` version you've used)
- A working API serving your prediction model based on `FastAPI`
- Notebooks and/or plots to support your decision process
- A `README.md` describing your approach
- A `requirements.txt` with the minimal set of dependencies necessary to run your api
- A `requirements-eda.txt` with all dependencies necessary to run your api and all additional notebooks or files that you may provide

You can basically make whatever choice you see fit, but we do expect you to justify all the decisions that you make.
We are not going to evaluate your test based on the accuracy of your model, but we do expect you to provide a critical discussion of the strength and weaknesses of your approach, and to elaborate on possible ways to improve your work.

**What's important for us**

- You must provide us with the necessary information to create a working environment to run your code.
- Your API must be able to run and respond to requests without error.
- It is very important that you guide us through your thought process, that you motivate your choices, and that you discuss the strengths and weaknesses of your approach and possible ways to improve your predictions. 

**What's not (so) important for us**

- You can use whatever other external software libraries you think are appropriate: pandas/numpy/scikit-learn are encouraged!
- The preprocessing/algorithms/loss functions and the split between train/validation/test set are yours to decide.
- You do not necessarily have to use all the data if you feel like some of it is irrelevant or not useful. 
- We are not going to penalize you for accuracy.

# The Weapons we provide you

In the `data` folder, you will find two `csv` files containing anonymized data from real bank accounts

- The `accounts.csv` contains a list of bank accounts, with the date of the last update of their financial data, and the balance on the account on the update date.
- The `transactions.csv` contains all available transactions on the accounts up to the update date, with an amount (in EUR) and the date of the day they were recorded.

> **Tip 2.** The available transaction history on each account is defined as the time elapsed since the oldest transaction recorded on this account and the **update date** of the account, not the date of the latest transaction on the account. If the date of the latest transaction on the account is older than the update date, that simply means that no transactions were recorded on the account after it up to the update date.

> **Tip 3.** Be careful that the available transaction history on the accounts generally does not span the whole lifespan of the accounts all the way up to their opening date. The initial balance at the date of the oldest transaction provided on a given account is not provided in the data, and it cannot be assumed to be zero! Only the final balance at the end of the available transaction history (on the update date) is known and provided in the `accounts.csv`. However, by combining the transactions and accounts data, you should be able to reverse the balance of the account back through time (back to the oldest transaction date for the account). This information might be useful as a feature of your predicting model.

# FastAPI 101

Provided with this repo is also a `main.py` file with a minimal [FastAPI](https://fastapi.tiangolo.com/) app template. Once you have installed the `requirements.txt` in your `Python` environment you will be able to run the main file by simply calling
`uvicorn main:app` inside your directory. This should start the local server and you should be able to see the automatically generated API docs at `http://127.0.0.1:8000/docs`.

In order to serve your model in the API, move your preprocessing to a function which you can call any input on. You do not have to worry about validating the inputs, FastAPI will do this for you! You can then move your predict functionality to the `predict` function in the `main.py` and return the predicted amount.

You can test your API using the `test_main.py` file, just make sure you are running the server by calling `uvicorn main:app` in another terminal window.

If you use `pandas`, you can convert the `transactions` of type `List[Transaction]` passed to the API to a `pd.DataFrame` by calling:

```python
import pandas as pd

df = pd.DataFrame(map(dict, transactions))
```

This is because the objects passed to the API are using `pydantic`'s `BaseModel` class which allows easy conversion from object to dictionary through the default `.dict()` implementation.

If you wish to learn more about how to use `FastAPI`:

- [Official FastAPI Docs](https://fastapi.tiangolo.com/)
- [Official pydantic Docs](https://pydantic-docs.helpmanual.io/)
- [Medium Post: How to Deploy a Machine Learning Model](https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-dc51200fe8cf)
