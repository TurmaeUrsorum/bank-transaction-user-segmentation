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
    skew_fix,
    handle_outliers,
    robust_scaler,
    standar_scaler_numeric_proses,
    label_encoder_categorical_proses,
    final_dataset
)
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import pytest


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


def test_skew_fix():
    df = pd.DataFrame(
        {
            "TransactionAmount": [100, 200, 300],
            "AccountBalance": [500, 600, 700],
            "AmountBalanceRatio": [0.2, 0.33, 0.43],
        }
    )
    params = {"method": "yeo-johnson"}

    result = skew_fix(df, params=params)

    # pastikan kolom baru ada
    assert "TransactionAmount_log" in result.columns
    assert "AmountBalanceRatio_yj" in result.columns

    # pastikan kolom lama hilang
    assert "TransactionAmount" not in result.columns
    assert "AmountBalanceRatio" not in result.columns

    # cek log transform-nya benar
    expected_log = np.log1p(df["TransactionAmount"])
    np.testing.assert_array_almost_equal(result["TransactionAmount_log"], expected_log)

    # cek power transform-nya jalan
    pt = PowerTransformer(method="yeo-johnson")
    expected_yj = pt.fit_transform(df[["AmountBalanceRatio"]]).flatten()
    np.testing.assert_array_almost_equal(result["AmountBalanceRatio_yj"], expected_yj)


@pytest.fixture
def dataframe_for_handling_outliers():
    return pd.DataFrame(
        {
            "TransactionAmount_log": [100, 200, 300, 10000],
            "AmountBalanceRatio_yj": [25, 30, 35, 3000000],
        }
    )


@pytest.mark.parametrize("method", ["capping", "remove"])
def test_handling_outliers(dataframe_for_handling_outliers, method):
    params = {
        "columns": ["TransactionAmount_log", "AmountBalanceRatio_yj"],
        "k": 1.5,
        "method": method,
    }

    result = handle_outliers(dataframe_for_handling_outliers, params)

    # tinggal assert sesuai metode
    if method == "capping":
        assert result.shape[0] == 4  # capping gak buang row
    elif method == "remove":
        assert result.shape[0] < 4  # remove bakal buang outlier


def test_robust_scaler():
    df = pd.DataFrame({"TransactionAmount_log": [100, 200, 300]})

    result = robust_scaler(df)

    assert "TransactionAmount_log_scaled" in result.columns
    assert "TransactionAmount_log" not in result.columns

    scaler = RobustScaler()
    expected_scaled = scaler.fit_transform(df[["TransactionAmount_log"]]).flatten()
    np.testing.assert_array_almost_equal(
        result["TransactionAmount_log_scaled"], expected_scaled
    )


def test_standar_scaler_numeric_proses():
    df = pd.DataFrame({"TransactionAmount_log": [100, 200, 300]})

    result = standar_scaler_numeric_proses(df)

    scaler = StandardScaler()
    expected_scaled = scaler.fit_transform(df[["TransactionAmount_log"]]).flatten()
    np.testing.assert_array_almost_equal(
        result["TransactionAmount_log"], expected_scaled
    )


def test_encoder_categorical_proses():
    # --- arrange ---
    df = pd.DataFrame({
        "TransactionType": ["DEBIT", "CREDIT", "CREDIT"],
        "Channel": ["ATM", "MOBILE", "WEB"],
        "Location": ["A", "B", "C"],
        "MerchantID": ["M1", "M2", "M3"],
    })

    # --- act ---
    result = label_encoder_categorical_proses(df)

    # --- assert ---

    # 1. Kolom Location dan MerchantID sudah dihapus
    assert "Location" not in result.columns
    assert "MerchantID" not in result.columns

    # 2. Pastikan kolom yang tersisa adalah kategorikal lain
    assert set(result.columns) == {"TransactionType", "Channel"}

    # 3. Pastikan hasil encoding sama dengan LabelEncoder bawaan
    le = LabelEncoder()
    expected_transaction_type = le.fit_transform(df["TransactionType"])
    np.testing.assert_array_equal(result["TransactionType"].values, expected_transaction_type) #type: ignore

    expected_channel = le.fit_transform(df["Channel"])
    np.testing.assert_array_equal(result["Channel"].values, expected_channel)# type: ignore

    # 4. Pastikan semua kolom hasil encoding bertipe int
    assert all(np.issubdtype(dtype, np.integer) for dtype in result.dtypes)

def test_final_dataset(monkeypatch):
    # --- arrange ---
    df = pd.DataFrame({
        "TransactionAmount": [100, 200, 300],
        "AccountBalance": [500, 600, 700],
        "TransactionType": ["DEBIT", "CREDIT", "CREDIT"],
        "Channel": ["ATM", "MOBILE", "WEB"],
        "Location": ["A", "B", "C"],
        "MerchantID": ["M1", "M2", "M3"],
    })

    # Mock fungsi scaler dan encoder biar gampang dites isolasinya
    def fake_scaler(df):
        return pd.DataFrame({
            "TransactionAmount": [0.0, 0.5, 1.0],
            "AccountBalance": [0.0, 0.5, 1.0],
        })

    def fake_encoder(df):
        return pd.DataFrame({
            "TransactionType": [0, 1, 1],
            "Channel": [0, 1, 2],
        })

    # --- act ---
    # Patch fungsi agar gak panggil fungsi asli
    import bank_transaction_user_segmentation.pipelines.data_preproses.nodes as nodes
    monkeypatch.setattr(nodes, "standar_scaler_numeric_proses", fake_scaler)
    monkeypatch.setattr(nodes, "label_encoder_categorical_proses", fake_encoder)

    result = final_dataset(df)

    # --- assert ---
    expected_cols = ["TransactionAmount", "AccountBalance", "TransactionType", "Channel"]
    assert list(result.columns) == expected_cols
    assert result.shape == (3, 4)
    assert result.isnull().sum().sum() == 0