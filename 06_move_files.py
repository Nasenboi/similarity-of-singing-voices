import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import pandas as pd
    from src.globals import DATASET_FOLDER
    from src.utils import get_onsets_es
    import librosa
    import matplotlib.pyplot as plt
    import shutil
    return DATASET_FOLDER, os, pd, shutil


@app.cell
def _(CSV_FOLDER, os, pd):
    subpath = "LargeDataset/triplet_selection"
    filename = "dataset_vq3_finished.csv"
    df = pd.read_csv(
        os.path.join(CSV_FOLDER, subpath, filename), index_col="track_id"
    )
    # mask = (df.voice_quality == 3) & (~df.multiple_voices)
    # df = df[mask]
    # .drop(columns="Unnamed: 0")
    df
    return (df,)


@app.cell
def _(df):
    df.iloc[0].song_path

    return


@app.cell
def _(df):
    df.iloc[0].vocal_path
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
