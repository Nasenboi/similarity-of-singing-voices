import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy
    import umap
    from sklearn.preprocessing import StandardScaler
    import numpy
    import os
    from dotenv import load_dotenv
    import librosa as lr
    import numpy as np

    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    load_dotenv(os.path.join(BASEDIR, ".env"))
    return StandardScaler, os, pd, umap


@app.cell
def _(os):
    RANDOM_STATE = 42
    DATASET_FOLDER = os.getenv("DATASET_FOLDER")
    CSV_FOLDER = os.getenv("CSV_FOLDER")
    AUDIO_FOLDER = os.path.join(DATASET_FOLDER, "fma_large")
    STEM_FOLDER = os.path.join(DATASET_FOLDER, "fma_large_stems")
    VOCALS_FILE_NAME = "vocals.mp3"
    return CSV_FOLDER, STEM_FOLDER, VOCALS_FILE_NAME


@app.cell
def _(CSV_FOLDER, os, pd):
    triplet_df = pd.read_csv(
        os.path.join(CSV_FOLDER, "triplet_df_voiced.csv"), index_col="track_id"
    )
    triplet_df
    return (triplet_df,)


@app.cell
def _(CSV_FOLDER, os, pd):
    features_df = pd.read_csv(
        os.path.join(CSV_FOLDER, "triplet_df_voiced_features_mfcc.csv"),
        index_col="track_id",
    )
    features_df
    return (features_df,)


@app.cell
def _(STEM_FOLDER, VOCALS_FILE_NAME, os):
    def getVocalPath(track_id: int):
        track_id_zp = "{0:06d}".format(track_id)
        vocal_path = os.path.join(
            STEM_FOLDER, track_id_zp[:3], track_id_zp, VOCALS_FILE_NAME
        )
        return vocal_path
    return (getVocalPath,)


@app.cell
def _(getVocalPath, triplet_df):
    triplet_df["vocal_audio_path"] = triplet_df.index.to_series().apply(
        getVocalPath
    )
    return


@app.cell
def _(StandardScaler, features_df):
    scaler = StandardScaler()
    voice_features_scaled = scaler.fit_transform(features_df)
    return (voice_features_scaled,)


@app.cell
def _(features_df, pd, umap, voice_features_scaled):
    reducer = umap.UMAP(
        n_components=2, n_neighbors=5, min_dist=0.1, metric="cosine"
    )
    umap_embeddings = reducer.fit_transform(voice_features_scaled)

    embedding_df = pd.DataFrame(
        umap_embeddings, index=features_df.index, columns=["UMAP1", "UMAP2"]
    )
    return (embedding_df,)


@app.cell
def _(embedding_df, triplet_df):
    triplet_df_umap = triplet_df.join(embedding_df)
    triplet_df_umap.dropna(subset=['UMAP1'])
    return


if __name__ == "__main__":
    app.run()
