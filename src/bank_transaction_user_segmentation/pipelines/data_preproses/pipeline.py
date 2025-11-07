"""
This is a boilerplate pipeline 'data_preproses'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa
from .nodes import feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=feature_engineering,
                inputs=["clean_bank_transaction", "params:feature_engineering"],
                outputs="feature_engineering",
                name="feature_engineering",
            )
        ]
    )
