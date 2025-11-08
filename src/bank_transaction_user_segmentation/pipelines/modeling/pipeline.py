"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa
from .nodes import modeling_kmedoids


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=modeling_kmedoids,
            inputs=["final_preprocessed_df", "params:modeling_cluster_params"],
            outputs="modeling_kmedoids",
            name="modeling_kmedoids",
        )
    ])
