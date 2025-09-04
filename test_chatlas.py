# In this script I am experimenting with Posit's python package, Chatlas

from chatlas import ChatOllama

chat = ChatOllama(
    model="gemma3:12b",
    system_prompt="You are a concise and helpful assistant Python assistant. Give me code only.",
)

chat.console()


# Write a python function that takes a data frame, and a response column, and
# returns the hat matrix for a simple regression model on all other variables.

# Chatlas's response via Gemma 12b:
import statsmodels.api as sm
import pandas as pd
import numpy as np


def calculate_hat_matrix(df, response_col):
    """
    Calculates the hat matrix for a linear regression model.

    Args:
        df (pd.DataFrame): The input DataFrame.
        response_col (str): The name of the response column.

    Returns:
        pd.DataFrame: The hat matrix.
    """
    # Isolate the predictor and response variables
    X = df.drop(columns=response_col)
    y = df[response_col]

    # Add a constant (intercept) to the predictor variables
    X = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(y, X)
    results = model.fit()

    # Get the hat matrix
    H = results.get_influence().hat_matrix_diag

    return pd.DataFrame(H)


# ChatGPT's response


def calculate_hat_matrix(df, response_col):
    """
    Calculates the hat matrix for a linear regression model.
    Args:
        df (pd.DataFrame): The input DataFrame.
        response_col (str): The name of the response column.
    Returns:
        pd.DataFrame: The hat matrix.
    """
    X = df.drop(columns=response_col)
    # y = df[response_col]
    X = sm.add_constant(X).to_numpy()  # convert to NumPy

    # Hat matrix: H = X (X'X)^(-1) X'
    hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
    return pd.DataFrame(hat_matrix)


# Example usage
data = {"x1": [1, 2, 3, 4, 5], "x2": [2, 2, 2, 2, 2], "y": [3, 5, 7, 9, 11]}
df = pd.DataFrame(data)
hat_mat = calculate_hat_matrix(df, "y")
print(hat_mat)
