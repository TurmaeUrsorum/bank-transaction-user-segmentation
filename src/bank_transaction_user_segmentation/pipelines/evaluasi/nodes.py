"""
This is a boilerplate pipeline 'evaluasi'
generated using Kedro 1.0.0
"""
import pandas as pd
import typing as tp

def cluster_summary_full_scaled(df, labels, cat_cols=None, num_cols=None) -> str:
    X = df.copy()
    X["Cluster"] = labels
    output_lines = []

    for cluster_id in sorted(X["Cluster"].unique()):
        cluster_data = X[X["Cluster"] == cluster_id]
        output_lines.append("="*60)
        output_lines.append(f"Cluster {cluster_id+1} (jumlah: {len(cluster_data)})")
        output_lines.append("="*60)
        
        # Numerik (scaled)
        if num_cols:
            output_lines.append("\n📊 Ringkasan fitur numerik (scaled):")
            desc = cluster_data[num_cols].describe().T[['mean','std','min','max']].round(2)
            output_lines.append(desc.to_string())
        
        # Kategorikal
        if cat_cols:
            output_lines.append("\n📂 Distribusi fitur kategorikal:")
            for col in cat_cols:
                dist = cluster_data[col].value_counts(normalize=True) * 100
                output_lines.append(f"\n{col}:")
                output_lines.append((dist.round(2).astype(str) + " %").to_string())

    # Gabungkan semua baris jadi satu string
    return "\n".join(output_lines)

def interpretasi_cluster(df: pd.DataFrame, params: tp.Dict) -> str:
    df = df.copy()

    df_hasil = cluster_summary_full_scaled(
        df=df,
        labels=df["cluster"],
        cat_cols=params["cat_cols"],
        num_cols=params["num_cols"],
    )

    return df_hasil