"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa
from .nodes import clean_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs="bank_transaction",
                outputs="clean_bank_transaction",
            )
        ]
    )
