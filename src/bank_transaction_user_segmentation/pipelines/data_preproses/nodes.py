"""
This is a boilerplate pipeline 'data_preproses'
generated using Kedro 1.0.0
"""

import pandas as pd
import typing as tp

# Feature Engineering


def feature_engineering(df: pd.DataFrame, params: tp.Dict) -> pd.DataFrame:
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
