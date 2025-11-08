"""
This is a boilerplate pipeline 'data_preproses'
generated using Kedro 1.0.0
"""

from sklearn.preprocessing import (
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    LabelEncoder,
)
import pandas as pd
import typing as tp
import numpy as np
import logging

# Feature Engineering

log = logging.getLogger(__name__)


def feature_engineering(df: pd.DataFrame, params: tp.Dict) -> pd.DataFrame:
    df = df.copy()
    # Rasio Amount/Balance (hindari division by zero)
    df["AmountBalanceRatio"] = df["TransactionAmount"] / (df["AccountBalance"] + 1e-6)

    # Prefensi Merchant & Location (Top-N encoding)

    # Top-N Merchant
    top_merchants = df["MerchantID"].value_counts().head(params["N"]).index
    df["MerchantTopN"] = df["MerchantID"].apply(
        lambda x: x if x in top_merchants else "OTHER"
    )

    # Top-N Location
    top_locations = df["Location"].value_counts().head(params["N"]).index
    df["LocationTopN"] = df["Location"].apply(
        lambda x: x if x in top_locations else "OTHER"
    )

    return df


def skew_fix(df: pd.DataFrame, params: tp.Dict) -> pd.DataFrame:
    df = df.copy()

    # fitur transaction Amount
    df["TransactionAmount_log"] = np.log1p(df["TransactionAmount"])

    # TransactionAmount masih ada
    pt = PowerTransformer(method=params["method"])
    df["AmountBalanceRatio_yj"] = pt.fit_transform(df[["AmountBalanceRatio"]])

    df = df.drop(["TransactionAmount", "AmountBalanceRatio"], axis=1)

    log.info(f"jumlah column: {df.columns}")

    return df


def handle_outliers_iqr(
    df: pd.DataFrame, columns: tp.List, k: float = 1.5, method: str = "remove"
) -> pd.DataFrame:
    df_copy = df.copy()
    mask = pd.Series(False, index=df.index)  # semua False dulu

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        if method == "capping":
            # Ganti nilai outlier dengan batas bawah/atas
            df_copy[col] = np.where(
                df[col] < lower, lower, np.where(df[col] > upper, upper, df[col])
            )

        else:
            # Tandai outlier untuk detect/remove
            mask |= (df[col] < lower) | (df[col] > upper)

    if method == "detect":
        return df[mask]  # hanya outliers
    elif method == "remove":
        return df[~mask]  # data tanpa outliers
    elif method == "capping":
        return df_copy  # data dengan capping
    else:
        raise ValueError("method harus salah satu dari: 'detect', 'remove', 'capping'")


def handle_outliers(df: pd.DataFrame, params: tp.Dict) -> pd.DataFrame:
    columns = params["columns"]
    df = handle_outliers_iqr(
        df, columns=columns, k=params["k"], method=params["method"]
    )
    return df


def robust_scaler(df: pd.DataFrame) -> pd.DataFrame:
    scaler = RobustScaler()
    df = df.copy()
    df["TransactionAmount_log_scaled"] = scaler.fit_transform(
        df[["TransactionAmount_log"]]
    )
    df = df.drop(["TransactionAmount_log"], axis=1)
    log.info(f"hasil robust scaler: {df.columns}")
    return df


def standar_scaler_numeric_proses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df_numeric = df.select_dtypes(include=["float64", "int64"])
    scaler = StandardScaler()
    df_numeric_scaled = scaler.fit_transform(df_numeric)
    df_numeric_scaled = pd.DataFrame(df_numeric_scaled, columns=df_numeric.columns)
    return df_numeric_scaled


def label_encoder_categorical_proses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df_categorical = df.select_dtypes(include=["object"])
    df_categorical = df_categorical.drop(columns=["Location", "MerchantID"])
    le = LabelEncoder()
    df_categorical_encoded = df_categorical.apply(lambda col: le.fit_transform(col))  # type: ignore
    df_categorical_encoded = pd.DataFrame(
        df_categorical_encoded, columns=df_categorical.columns
    )
    return df_categorical_encoded


def final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_numeric = standar_scaler_numeric_proses(df)
    df_categorical = label_encoder_categorical_proses(df)
    df_final = pd.concat([df_numeric, df_categorical], axis=1)
    return df_final
