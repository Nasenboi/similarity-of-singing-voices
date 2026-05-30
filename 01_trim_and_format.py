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
    # Initial Imports
    import os
    import sys

    import marimo as mo
    import numpy as np
    import pandas as pd

    # utils.py file
    # in: FMA: A Dataset For Music Analysis
    # Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset for Music Analysis. In 18th International Society for Music Information Retrieval Conference (ISMIR).
    # available under "https://github.com/mdeff/fma"
    from src.FMA.utils import get_audio_path, load
    from src.globals import AUDIO_FOLDER, CSV_FOLDER, STEMS_FOLDER, TRACKS_PATH, UVR_MODEL_PATH

    return (
        AUDIO_FOLDER,
        CSV_FOLDER,
        STEMS_FOLDER,
        TRACKS_PATH,
        UVR_MODEL_PATH,
        get_audio_path,
        load,
        mo,
        os,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load the FMA Dataset using their utils
    """)
    return


@app.cell
def _(AUDIO_FOLDER, TRACKS_PATH, get_audio_path, load):
    tracks_df = load(TRACKS_PATH)
    tracks_df["song_path"] = [get_audio_path(AUDIO_FOLDER, i) for i in tracks_df.index]
    tracks_df
    return (tracks_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Format the Dataset
    Create a smaller dataset with relevant track information
    """)
    return


@app.cell
def _(pd, tracks_df):
    fma = pd.DataFrame(
        {
            "track_id": tracks_df.index,
            "genre_top": tracks_df["track", "genre_top"],
            "artist": tracks_df["artist", "name"],
            "artist_id": tracks_df["artist", "id"],
            "members": tracks_df["artist", "members"],
            "location": tracks_df["artist", "location"],
            "album": tracks_df["album", "title"],
            "creation_date": tracks_df["track", "date_created"],
            "release_date": tracks_df["album", "date_released"],
            "song_path": tracks_df["song_path"],
            "artist_website": tracks_df["artist", "website"],
            "license": tracks_df["track", "license"],
            "publisher": tracks_df["track", "publisher"],
            "lyricist": tracks_df["track", "lyricist"],
            "title": tracks_df["track", "title"],
        }
    )
    fma.set_index("track_id", inplace=True)
    fma
    return (fma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trim the dataset

    1. Ignore genres where singing voices are less likely
    2. Use only one track per artist name
    """)
    return


@app.cell
def _(fma):
    genres_to_ignore = [
        "Electronic",
        "Experimental",
        "Instrumental",
        "Classical",
        "Hip-Hop",
        "Jazz",
        "Old-Time / Historic",
    ]
    ignore_genre_mask = ~fma["genre_top"].isin(genres_to_ignore)
    fma_genres = fma[ignore_genre_mask]

    fma_genres
    return (fma_genres,)


@app.cell
def _(fma_genres):
    all_artists = fma_genres.artist.unique().copy()
    fma_single_file = fma_genres.drop_duplicates(subset="artist", keep="last")
    fma_single_file.loc[:, "artist_overlaps"] = fma_single_file["artist"].apply(
        lambda a: [x for x in all_artists if str(a) in x and x != a]
    )
    # remove overlaps
    # fma_single_file = fma_single_file[~fma_single_file["artist_overlaps"].astype(bool)]
    fma_single_file
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create Vocal and Instrument Stems

    Separator class from [GitHub](https://github.com/nomadkaraoke/python-audio-separator), thanks to "nomadkaraoke".
    Separator model from [HuggingFace](https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt) thanks to Kimberley Jensen

    Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source Separation. https://arxiv.org/abs/2310.01809
    """)
    return


@app.cell
def _(STEMS_FOLDER, UVR_MODEL_PATH):
    import logging
    import shutil

    from audio_separator.separator import Separator

    UVR_MODEL_NAME = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    OUTPUT_FORMAT = "MP3"

    separator = Separator(
        model_file_dir=UVR_MODEL_PATH,
        output_dir=STEMS_FOLDER,
        output_format=OUTPUT_FORMAT,
        log_level=logging.WARNING,
    )

    separator.load_model(model_filename=UVR_MODEL_NAME)
    return separator, shutil


@app.cell
def _(STEMS_FOLDER, fma_reduced, os, separator, shutil):
    def processFile(track_id: str) -> str:
        """Process a single file

        Args:
            track_id (str): The dataset id for the audio file to process

        Returns:
            str | None: Returns the file path to the vocal stem if method ran successfully
        """
        try:
            track_id_zp = "{0:06d}".format(track_id)
            sub_path = os.path.join(track_id_zp[:3], track_id_zp)
            os.makedirs(os.path.join(STEMS_FOLDER, sub_path), exist_ok=True)
            output_names = {
                "Vocals": os.path.join(sub_path, "vocals"),
                "Instrumental": os.path.join(sub_path, "instrumental"),
            }
            vocals_path = os.path.join(STEMS_FOLDER, f"{output_names['Vocals']}.mp3")
            if os.path.exists(vocals_path):
                return vocals_path
            separator.separate(fma_reduced.loc[track_id].song_path, output_names)
            for name in ["Vocals", "Instrumental"]:
                false_path = os.path.join(STEMS_FOLDER, f"{output_names[name].replace('/', '_')}.mp3")
                true_path = os.path.join(STEMS_FOLDER, f"{output_names[name]}.mp3")
                shutil.move(false_path, true_path)
        except Exception as e:
            print(e)
            return None

    return (processFile,)


@app.cell
def _(fma_reduced, mo, processFile):
    fma_reduced["vocal_path"] = [
        processFile(int(track_id))
        for track_id in mo.status.progress_bar(
            fma_reduced.index,
            title="Separating vocals...",
            subtitle="Processing tracks",
            completion_title="Done!",
            completion_subtitle=f"Processed {len(fma_reduced)} tracks",
        )
    ]

    fma_reduced.dropna(subset=["vocal_path"])
    return


@app.cell
def _(CSV_FOLDER, os, pd):
    track_df = pd.read_csv(
        os.path.join(
            CSV_FOLDER,
            "LargeDataset",
            "additional_features",
            "high_level_features.csv",
        ),
        index_col="track_id",
    )
    track_df
    return (track_df,)


@app.cell
def _(fma, track_df):
    track_df["artist_id"] = track_df.apply(lambda x: fma.loc[x.name, "artist_id"], axis=1)
    return


@app.cell
def _(track_df):
    len(track_df.artist_id.unique())
    return


@app.cell
def _(fma_genres, track_df):
    fma_reduced = fma_genres[fma_genres["artist_id"].isin(track_df["artist_id"])]
    fma_reduced
    return (fma_reduced,)


if __name__ == "__main__":
    app.run()
