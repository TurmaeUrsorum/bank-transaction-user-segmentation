"""
This is a boilerplate test file for pipeline 'data_cleaning'
generated using Kedro 1.0.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

from bank_transaction_user_segmentation.pipelines.data_cleaning.nodes import clean_data
import pytest
import pandas as pd


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    data = {
        "TransactionAmount": [100, 200, 200, None],
        "TransactionType": ["DEBIT", "CREDIT", "CREDIT", "DEBIT"],
        "Channel": ["ATM", "MOBILE", "MOBILE", "ATM"],
        "MerchantID": [1, 2, 2, 3],
        "Location": ["A", "B", "B", "C"],
        "AccountBalance": [500, 600, 600, 700],
        "CustomerAge": [25, 30, 30, 40],
        "CustomerOccupation": ["IT", "Banker", "Banker", "Doctor"],
        "TransactionDuration": [30, 40, 40, 50],
    }

    data_frame = pd.DataFrame(data)
    return data_frame


def test_clean_data_non_nan(sample_dataset: pd.DataFrame):
    result = clean_data(sample_dataset)
    assert result.isna().sum().sum() == 0

    expected_cols = [
        "TransactionAmount",
        "TransactionType",
        "Channel",
        "MerchantID",
        "Location",
        "AccountBalance",
        "CustomerAge",
        "CustomerOccupation",
        "TransactionDuration",
    ]

    assert list(result.columns) == expected_cols


def test_clean_data_non_duplicate(sample_dataset: pd.DataFrame):
    result = clean_data(sample_dataset)
    assert result.duplicated().sum() == 0

    expected_cols = [
        "TransactionAmount",
        "TransactionType",
        "Channel",
        "MerchantID",
        "Location",
        "AccountBalance",
        "CustomerAge",
        "CustomerOccupation",
        "TransactionDuration",
    ]

    assert list(result.columns) == expected_cols
