"""
This is a boilerplate pipeline 'data_preproses'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa
from .nodes import (
    feature_engineering,
    skew_fix,
    handle_outliers,
    robust_scaler,
    final_dataset,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=feature_engineering,
                inputs=["clean_bank_transaction", "params:feature_engineering"],
                outputs="feature_engineering",
                name="feature_engineering",
            ),
            node(
                func=skew_fix,
                inputs=["feature_engineering", "params:skew_fix"],
                outputs="skew_fix",
                name="skew_fix",
            ),
            node(
                func=handle_outliers,
                inputs=["skew_fix", "params:handle_outliers"],
                outputs="handle_outliers",
                name="handle_outliers",
            ),
            node(
                func=robust_scaler,
                inputs="handle_outliers",
                outputs="robust_scaler_df",
                name="robust_scaler_df",
            ),
            node(
                func=final_dataset,
                inputs="robust_scaler_df",
                outputs="final_preprocessed_df",
                name="final_preprocessed_df",
            ),
        ]
    )
