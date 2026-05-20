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
    # initial imports
    import marimo as mo
    import os
    import pandas as pd
    from src.globals import CSV_FOLDER
    from src.utils import get_onsets_es
    import librosa
    import matplotlib.pyplot as plt
    return CSV_FOLDER, get_onsets_es, librosa, mo, os, pd, plt


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
    df = pd.read_csv(
        os.path.join(CSV_FOLDER, subpath, filename), index_col="track_id"
    )
    # mask = (df.voice_quality == 3) & (~df.multiple_voices)
    # df = df[mask]
    # .drop(columns="Unnamed: 0")
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    # Calculate Onsets

    Onsets are calculated using Essentias [Onset detection](https://essentia.upf.edu/tutorial_rhythm_onsetdetection.html) algorithm.
    """)
    return


@app.cell
def _(df, get_onsets_es, librosa, plt):
    def getOnsets(row):
        try:
            # do not calculate for unvoiced or multiple voices
            if (not row["is_voiced"]) or row["multiple_voices"]:
                return None
            onsets = get_onsets_es(
                row["vocal_path"],
                silenceThreshold=0.1,
                delay=5,
                alpha=0.1,
                onset_pause=4,
                round_down=False,
            )
            if len(onsets) == 0:
                return None
            else:
                return onsets
        except Exception as e:
            return None


    idx = 0
    row = df.iloc[idx]
    y, sr = librosa.load(row["vocal_path"], mono=True)

    librosa.display.waveshow(y=y, sr=sr)

    plt.vlines(getOnsets(row), ymin=-1, ymax=1, color="r", alpha=0.7)
    plt.show()
    return (getOnsets,)


@app.cell
def _(df, getOnsets):
    df["onsets"] = df.apply(getOnsets, axis=1)
    df
    return


if __name__ == "__main__":
    app.run()
