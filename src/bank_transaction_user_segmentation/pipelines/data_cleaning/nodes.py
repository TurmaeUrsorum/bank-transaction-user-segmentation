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
    "TransactionDuration"
]
]