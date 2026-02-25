import marimo

__generated_with = "0.16.5"
app = marimo.App(
    width="medium",
    layout_file="layouts/triplet-preparation_part2.slides.json",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Prepare Triplets for the User Survey

    ### Initialize Project: Import libs and set globals
    """
    )
    return


@app.cell
def _():
    import os
    import shutil
    from os import getenv, path
    import marimo as mo
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from funasr import AutoModel

    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    load_dotenv(os.path.join(BASEDIR, ".env"))
    return AutoModel, getenv, mo, os, path, pd


@app.cell
def _(mo):
    mo.md(r"""# Load and Prepare Datasaet""")
    return


@app.cell
def _(getenv, path):
    RANDOM_STATE = 42
    DATASET_FOLDER = getenv("DATASET_FOLDER")
    CSV_FOLDER = getenv("CSV_FOLDER")
    AUDIO_FOLDER = path.join(DATASET_FOLDER, "fma_large")
    STEM_FOLDER = path.join(DATASET_FOLDER, "fma_large_stems")
    # GENRES_PATH = path.join(METADATA_FOLDER, "genres.csv")
    # FEATURES_PATH = path.join(METADATA_FOLDER, "features.csv")
    return CSV_FOLDER, STEM_FOLDER


@app.cell
def _(CSV_FOLDER, path, pd):
    triplet_df = pd.read_csv(path.join(CSV_FOLDER, "triplet_df.csv"), index_col="track_id")
    triplet_df
    return (triplet_df,)


@app.cell
def _(mo):
    mo.md(r"""# Check if Voice Stems Actually Cointain a Voice""")
    return


@app.cell
def _(AutoModel, STEM_FOLDER, getenv, os, path):
    VAD_MODEL_PATH = path.join(getenv("MODEL_FOLDER"), "VAD")
    VAD_MODEL_NAME = "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    VOCALS_FILE_NAME = "vocals.mp3"

    vad_model = AutoModel(
        model=path.join(VAD_MODEL_PATH, VAD_MODEL_NAME), disable_update=True
    )


    def getVoiceActivity(track_id: int):
        track_id_zp = "{0:06d}".format(track_id)
        vocal_path = os.path.join(
            STEM_FOLDER, track_id_zp[:3], track_id_zp, VOCALS_FILE_NAME
        )
        try:
            return vad_model.generate(input=vocal_path)[0]["value"]
        except Exception as e:
            return []
    return (getVoiceActivity,)


@app.cell
def _(getVoiceActivity, mo, triplet_df):
    triplet_df["is_voiced"] = None
    for track_id in mo.status.progress_bar(
        triplet_df.index,
        title="Analyzing vocals...",
        subtitle="Processing tracks",
        completion_title="Done!",
        completion_subtitle=f"Processed {len(triplet_df)} tracks",
    ):
        try:
            triplet_df.loc[track_id, "is_voiced"] = len(getVoiceActivity(int(track_id))) > 0 
        except Exception as e:
            print(e)
    return


@app.cell
def _(triplet_df):
    triplet_df[triplet_df["is_voiced"] == False]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
