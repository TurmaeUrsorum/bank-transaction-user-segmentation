"""
This is a boilerplate test file for pipeline 'evaluasi'
generated using Kedro 1.0.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from bank_transaction_user_segmentation.pipelines.evaluasi.nodes import cluster_summary_full_scaled, interpretasi_cluster
import pandas as pd
import pytest
from io import StringIO

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": [25, 30, 22, 40, 35],
        "Spending": [100, 200, 150, 300, 250],
        "Channel": ["ATM", "WEB", "WEB", "ATM", "WEB"]
    })

@pytest.fixture
def sample_labels():
    return [0, 0, 1, 1, 0]

def test_cluster_summary_full_scaled_basic(sample_df, sample_labels):
    result = cluster_summary_full_scaled(
        sample_df,
        labels=sample_labels,
        cat_cols=["Channel"],
        num_cols=["Age", "Spending"],
    )

    # --- ASSERTS ---

    # 1. Output tipe string
    assert isinstance(result, str)

    # 2. Harus mengandung nama cluster
    assert "Cluster 1" in result
    assert "Cluster 2" in result

    # 3. Harus mengandung ringkasan numerik
    assert "Ringkasan fitur numerik" in result

    # 4. Harus mengandung distribusi kategorikal
    assert "Distribusi fitur kategorikal" in result
    assert "Channel" in result

    # 5. Format garis pemisah
    assert "=" * 60 in result

@pytest.fixture
def sample_df2():
    return pd.DataFrame({
        "Age": [25, 30, 22, 40, 35],
        "Spending": [100, 200, 150, 300, 250],
        "Channel": ["ATM", "WEB", "WEB", "ATM", "WEB"],
        "cluster": [0, 0, 1, 1, 0]
    })

@pytest.fixture
def sample_params():
    return {
        "cat_cols": ["Channel"],
        "num_cols": ["Age", "Spending"]
    }

def test_interpretasi_cluster_returns_string(sample_df2, sample_params):
    result = interpretasi_cluster(sample_df2, sample_params)

    # 1 hasilnya string
    assert isinstance(result, str)

    # 2 ada teks cluster
    assert "Cluster 1" in result
    assert "Cluster 2" in result

    # 3 pastikan ringkasan numerik dan kategorikal muncul
    assert "Ringkasan fitur numerik" in result
    assert "Distribusi fitur kategorikal" in result
