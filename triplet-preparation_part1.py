import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/triplet-preparation_part1.slides.json",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prepare Triplets for the User Survey

    ### Initialize Project: Import libs and set globals
    """)
    return


@app.cell
def _():
    import os
    import shutil
    from os import getenv, path
    from random import choice
    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests
    import torch
    import torch.nn as nn
    import umap
    from dotenv import load_dotenv
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    from transformers import Data2VecAudioModel, Wav2Vec2Processor
    from audio_separator.separator import Separator
    import utils
    import logging

    load_dotenv()
    return choice, getenv, mo, os, path, pd, shutil, utils


@app.cell
def _(mo):
    mo.md(r"""
    # Load and Prepare Datasaet
    """)
    return


@app.cell
def _(getenv, path):
    RANDOM_STATE = 42
    DATASET_FOLDER = getenv("DATASET_FOLDER")
    METADATA_FOLDER = path.join(DATASET_FOLDER, "fma_metadata")
    AUDIO_FOLDER = path.join(DATASET_FOLDER, "fma_large")
    TRACKS_PATH = path.join(METADATA_FOLDER, "tracks.csv")
    # GENRES_PATH = path.join(METADATA_FOLDER, "genres.csv")
    # FEATURES_PATH = path.join(METADATA_FOLDER, "features.csv")
    return AUDIO_FOLDER, TRACKS_PATH


@app.cell
def _(TRACKS_PATH, utils):
    tracks_df = utils.load(TRACKS_PATH)
    tracks_df = tracks_df[tracks_df["set", "subset"] <= "small"]
    return (tracks_df,)


@app.cell
def _(tracks_df):
    tracks_df
    return


@app.cell
def _(tracks_df):
    tracks_df["track", "genre_top"]
    return


@app.cell
def _(AUDIO_FOLDER, tracks_df, utils):
    tracks_df["audio_path"] = [
        utils.get_audio_path(AUDIO_FOLDER, i) for i in tracks_df.index
    ]
    return


@app.cell
def _():
    # sum([path.exists(p) for p in tracks_df["audio_path"]] * 1)
    return


@app.cell
def _(choice, mo, tracks_df):
    audio_filename = choice(tracks_df["audio_path"].tolist())
    mo.audio(audio_filename)
    return


@app.cell
def _(pd, tracks_df):
    triplet_df = pd.DataFrame(
        {
            "track_id": tracks_df.index,
            "genre_top": tracks_df["track", "genre_top"],
            "artist": tracks_df["artist", "name"],
            "album": tracks_df["album", "title"],
            "creation_date": tracks_df["track", "date_created"],
            "release_date": tracks_df["album", "date_released"],
            "audio_path": tracks_df["audio_path"],
        }
    )

    triplet_df["filename"] = (
        triplet_df["track_id"].astype(str).str.zfill(6) + ".mp3"
    )
    triplet_df["folder"] = triplet_df["filename"].str[:3]

    triplet_df.set_index("track_id", inplace=True)

    genres_to_ignore = ["Electronic", "Experimental", "Instrumental"]
    ignore_genre_mask = ~triplet_df["genre_top"].isin(genres_to_ignore)
    triplet_df = triplet_df[ignore_genre_mask]

    triplet_df
    return (triplet_df,)


@app.cell
def _(mo):
    mo.md(r"""
    # Create Voice and Instrument Stems
    """)
    return


app._unparsable_cell(
    r"""
    UVR_MODEL_PATH = path.join(getenv(\"MODEL_FOLDER\"), \"UVR\")
    UVR_MODEL_NAME = \"model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt\"
    STEM_PATH = path.join(DATASET_FOLDER, \"fma_large_stems\")
    OUTPUT_FORMAT = \"MP3\"

    #separator = Separator(
        model_file_dir=UVR_MODEL_PATH,
        output_dir=STEM_PATH,
        output_format=OUTPUT_FORMAT,
        log_level=logging.WARNING,
    )

    #separator.load_model(model_filename=UVR_MODEL_NAME)
    """,
    name="_"
)


@app.cell
def _(STEM_PATH, os, path, shutil):
    def processFile(track_id: str, folder: str, audio_path: str):
        track_id_zp = "{0:06d}".format(track_id)
        sub_path = path.join(track_id_zp[:3], track_id_zp)
        os.makedirs(path.join(STEM_PATH, sub_path), exist_ok=True)
        output_names = {
            "Vocals": path.join(sub_path, "vocals"),
            "Instrumental": path.join(sub_path, "instrumental"),
        }
        #separator.separate(audio_path, output_names)
        for name in ["Vocals", "Instrumental"]:
            false_path = path.join(
                STEM_PATH, f"{output_names[name].replace('/', '_')}.mp3"
            )
            true_path = path.join(STEM_PATH, f"{output_names[name]}.mp3")
            shutil.move(false_path, true_path)
    return


@app.cell
def _(mo, triplet_df):
    for track_id in mo.status.progress_bar(
        triplet_df.index,
        title="Separating vocals...",
        subtitle="Processing tracks",
        completion_title="Done!",
        completion_subtitle=f"Processed {len(triplet_df)} tracks",
    ):
        row = triplet_df.loc[track_id]
        try:
            #processFile(int(track_id), row["folder"], row["audio_path"])
            break
        except Exception as e:
            print(e)
    return


if __name__ == "__main__":
    app.run()
