import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Import Python Packages
    """)
    return


@app.cell
def _():
    import os
    import shutil

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.globals import DATASET_FOLDER
    from src.utils import get_onsets_es

    return DATASET_FOLDER, mo, os, pd, shutil


@app.cell
def _(mo):
    mo.md(r"""
    # Load the Dataset
    """)
    return


@app.cell
def _(CSV_FOLDER, os, pd):
    subpath = "LargeDataset/triplet_selection"
    filename = "dataset_vq3_finished.csv"
    df = pd.read_csv(os.path.join(CSV_FOLDER, subpath, filename), index_col="track_id")
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    # Move Relecant Files to "Release" Folder
    """)
    return


@app.cell
def _(DATASET_FOLDER, os, shutil):
    OLD_PATH = DATASET_FOLDER
    NEW_PATH = DATASET_FOLDER.replace("MusicVoiceCluster", "Release")

    def moveFiles(row):
        old_song_path = row["song_path"]
        old_vocal_path = row["vocal_path"]
        new_song_path = old_song_path.replace(OLD_PATH, NEW_PATH)
        new_vocal_path = old_vocal_path.replace(OLD_PATH, NEW_PATH)

        os.makedirs(os.path.dirname(new_song_path), exist_ok=True)
        os.makedirs(os.path.dirname(new_vocal_path), exist_ok=True)
        shutil.move(old_song_path, new_song_path)
        shutil.move(old_vocal_path, new_vocal_path)

    return (moveFiles,)


@app.cell
def _(df, moveFiles):
    df.apply(moveFiles, axis=1)
    return


if __name__ == "__main__":
    app.run()
