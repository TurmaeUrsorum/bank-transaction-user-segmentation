"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 1.0.0
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_main_features = df[
        [
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
    ]

    df_main_features = df_main_features.dropna()

    df_main_features = df_main_features.drop_duplicates()

    return df_main_features
