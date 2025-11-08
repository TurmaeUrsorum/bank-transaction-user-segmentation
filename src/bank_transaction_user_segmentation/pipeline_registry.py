"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from bank_transaction_user_segmentation.pipelines.data_cleaning import pipeline as dc
from bank_transaction_user_segmentation.pipelines.modeling import pipeline as md
from bank_transaction_user_segmentation.pipelines.evaluasi import pipeline as ev
from bank_transaction_user_segmentation.pipelines.data_preproses import pipeline as dp


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    data_cleaning_pipeline = dc.create_pipeline()
    data_preproses_pipeline = dp.create_pipeline()
    modeling_pipeline = md.create_pipeline()
    evaluasi_pipeline = ev.create_pipeline()

    return {
        "__default__": data_cleaning_pipeline
        + data_preproses_pipeline
        + modeling_pipeline
        + evaluasi_pipeline,
        "dc": data_cleaning_pipeline,
        "dp": data_preproses_pipeline,
        "md": modeling_pipeline,
        "ev": evaluasi_pipeline,
    }
