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
    return lr, mo, os, pd


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
        os.path.join(CSV_FOLDER, "triplet_df_checked5.csv"), index_col="track_id"
    )
    triplet_df
    return (triplet_df,)


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
def _(triplet_df):
    if triplet_df["is_voiced_check"].eq(False).any():
        first_unchecked = triplet_df[triplet_df["is_voiced_check"] == False].index[
            0
        ]
        first_unchecked = triplet_df.index.get_loc(first_unchecked)
    else:
        first_unchecked = 0
    first_unchecked
    return (first_unchecked,)


@app.cell
def _(first_unchecked, mo, triplet_df):
    get_idx, set_idx = mo.state(first_unchecked)


    def setCurrentIdx(step: int):
        set_idx(lambda value: max(0, min(len(triplet_df) - 1, value + step)))


    def answerAndGoOn(isVoiced):
        track_id = triplet_df.iloc[get_idx()].name
        triplet_df.loc[track_id, "is_voiced"] = isVoiced
        triplet_df.loc[track_id, "is_voiced_check"] = True
        setCurrentIdx(1)


    increment_button = mo.ui.button(
        on_click=lambda value: setCurrentIdx(1), label="+"
    )
    decrement_button = mo.ui.button(
        on_click=lambda value: setCurrentIdx(-1), label="-"
    )

    voiced_true_button = mo.ui.button(
        on_click=lambda value: answerAndGoOn(True), label="Yes"
    )

    voiced_false_button = mo.ui.button(
        on_click=lambda value: answerAndGoOn(False), label="No"
    )

    form = (
        mo.md(
            """
            **Navigation**

            {decbtn} {incbtn}

            **Voiced?**

            {falsebtn} {truebtn}
            """
        )
        .batch(
            decbtn=decrement_button,
            incbtn=increment_button,
            falsebtn=voiced_false_button,
            truebtn=voiced_true_button,
        )
        .form()
    )

    form
    return (get_idx,)


@app.cell
def _(get_idx, lr, mo, triplet_df):
    checked_count = triplet_df["is_voiced_check"].sum()
    max_count = len(triplet_df)
    perc_done = round(100 * (checked_count / max_count), 1)

    row = triplet_df.iloc[get_idx()]
    current_path = row.vocal_audio_path

    y, sr = lr.load(current_path)

    print("Progress :", f"{checked_count}/{max_count} ({perc_done}%)")
    print("Index    :", row.name)
    print("Artist   :", row.artist)
    print("Genre    :", row.genre_top)
    print("IsVoiced :", row.is_voiced)
    print("Checked  :", row.is_voiced_check)


    mo.audio(src=y, rate=sr * 1.5)
    return


@app.cell
def _(triplet_df):
    triplet_df
    return


if __name__ == "__main__":
    app.run()
