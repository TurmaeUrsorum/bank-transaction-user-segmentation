"""
This is a boilerplate test file for pipeline 'modeling'
generated using Kedro 1.0.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

from bank_transaction_user_segmentation.pipelines.modeling.nodes import (
    modeling_kmedoids,
)
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "TransactionAmount": [100, 200, 150, 300, 250],
            "CustomerAge": [25, 30, 22, 40, 35],
            "Channel": ["ATM", "MOBILE", "WEB", "ATM", "WEB"],
        }
    )


@pytest.fixture
def kmedoid_params():
    return {
        "cluster_k": 2,
        "metric": "precomputed",
        "random_state": 42,
    }


def test_modeling_kmedoids(sample_df, kmedoid_params):
    result = modeling_kmedoids(sample_df, kmedoid_params)

    assert "cluster" in result.columns

    assert result.shape[0] == sample_df.shape[0]

    assert result["cluster"].isna().sum() == 0

    unique_clusters = result["cluster"].unique()
    assert all(c in range(kmedoid_params["cluster_k"]) for c in unique_clusters)
