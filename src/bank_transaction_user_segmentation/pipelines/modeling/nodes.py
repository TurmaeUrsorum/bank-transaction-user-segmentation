"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 1.0.0
"""
from sklearn_extra.cluster import KMedoids
import gower
import pandas as pd
import typing as tp

def modeling_kmedoids(df: pd.DataFrame, params: tp.Dict) -> pd.DataFrame:
    gower_dist = gower.gower_matrix(df)

    k = params["cluster_k"]
    pam = KMedoids(n_clusters=params["cluster_k"], metric=params["metric"], random_state=params["random_state"])
    labels = pam.fit_predict(gower_dist)

    df["cluster"] = labels

    return df