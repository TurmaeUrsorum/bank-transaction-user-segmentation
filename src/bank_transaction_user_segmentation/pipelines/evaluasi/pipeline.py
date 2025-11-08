"""
This is a boilerplate pipeline 'evaluasi'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node # noqa
from .nodes import interpretasi_cluster

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=interpretasi_cluster,
            inputs=["modeling_kmedoids", "params:interpretasi_cluster"],
            outputs="interpretasi_cluster",
            name="interpretasi_cluster",
        )
    ])
