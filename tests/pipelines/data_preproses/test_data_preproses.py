"""
This is a boilerplate test file for pipeline 'data_preproses'
generated using Kedro 1.0.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

from bank_transaction_user_segmentation.pipelines.data_preproses.nodes import (
    feature_engineering,
)
import pandas as pd


def test_feature_engineering():
    # Data dummy
    data = {
        "TransactionAmount": [100, 200, 50, 300, 200],
        "AccountBalance": [1000, 2000, 500, 0, 3000],
        "MerchantID": ["A", "A", "B", "C", "C"],
        "Location": ["X", "Y", "X", "Z", "Z"],
    }
    df = pd.DataFrame(data)

    # Parameter: ambil 5merchant & location paling sering
    params = {"N": 2}

    result = feature_engineering(df.copy(), params=params)

    # Testing pertama : jumlah column harus sama
    expected_columns = [
        "TransactionAmount",
        "AccountBalance",
        "MerchantID",
        "Location",
        "AmountBalanceRatio",
        "MerchantTopN",
        "LocationTopN",
    ]

    for col in expected_columns:
        assert col in result.columns, f"{col} tidak ditemukan"

    # testing kedua: expect ratio

    expected_ratio = df["TransactionAmount"] / (df["AccountBalance"] + 1e-6)
    expected_ratio.name = "AmountBalanceRatio"
    pd.testing.assert_series_equal(result["AmountBalanceRatio"], expected_ratio)

    assert all(
        result[result["MerchantID"].isin(["A", "C"])]["MerchantTopN"].isin(["A", "C"])
    )
    # Merchant B hanya muncul sekali → harus jadi OTHER
    assert (result.loc[result["MerchantID"] == "B", "MerchantTopN"] == "OTHER").all()

    assert all(
        result[result["Location"].isin(["Z", "X"])]["LocationTopN"].isin(["Z", "X"])
    )
    # Location Y cuma sekali → jadi OTHER
    assert (result.loc[result["Location"] == "Y", "LocationTopN"] == "OTHER").all()
