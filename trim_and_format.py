import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    # Initial Imports
    import marimo as mo
    import pandas as pd
    import numpy as np
    import os
    import sys

    # utils.py file
    # in: FMA: A Dataset For Music Analysis
    # Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset for Music Analysis. In 18th International Society for Music Information Retrieval Conference (ISMIR).
    # available under "https://github.com/mdeff/fma"
    from src.FMA.utils import load, get_audio_path
    from src.globals import (
        CSV_FOLDER,
        TRACKS_PATH,
        AUDIO_FOLDER,
        STEMS_FOLDER,
        UVR_MODEL_PATH,
    )
    return (
        AUDIO_FOLDER,
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
    tracks_df["song_path"] = [
        get_audio_path(AUDIO_FOLDER, i) for i in tracks_df.index
    ]
    tracks_df
    return (tracks_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Format the Dataset
    Create a minimalist dataset with essential track information
    """)
    return


@app.cell
def _(pd, tracks_df):
    fma = pd.DataFrame(
        {
            "track_id": tracks_df.index,
            "genre_top": tracks_df["track", "genre_top"],
            "artist": tracks_df["artist", "name"],
            "album": tracks_df["album", "title"],
            "creation_date": tracks_df["track", "date_created"],
            "release_date": tracks_df["album", "date_released"],
            "song_path": tracks_df["song_path"],
        }
    )
    fma.set_index("track_id", inplace=True)
    fma
    return (fma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trim the dataset
    Remove all rows which will most likely not contain any clean singing voice parts.
    Use only one track per artist. Mark overlapping artist names.
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
    return (fma_single_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create Vocal and Instrument Stems
    """)
    return


@app.cell
def _(STEM_PATH, os, path, separator, shutil, triplet_df_single):
    def processFile(track_id: str) -> str:
        """Process a single file

        Args:
            track_id (str): The dataset id for the audio file to process

        Returns:
            str | None: Returns the file path to the vocal stem if method ran successfully
        """
        try:
            track_id_zp = "{0:06d}".format(track_id)
            sub_path = path.join(track_id_zp[:3], track_id_zp)
            os.makedirs(path.join(STEM_PATH, sub_path), exist_ok=True)
            output_names = {
                "Vocals": path.join(sub_path, "vocals"),
                "Instrumental": path.join(sub_path, "instrumental"),
            }
            vocals_path = path.join(STEM_PATH, f"{output_names['Vocals']}.mp3")
            if os.path.exists(vocals_path):
                return vocals_path
            separator.separate(
                triplet_df_single.loc[track_id].song_path, output_names
            )
            for name in ["Vocals", "Instrumental"]:
                false_path = path.join(
                    STEM_PATH, f"{output_names[name].replace('/', '_')}.mp3"
                )
                true_path = path.join(STEM_PATH, f"{output_names[name]}.mp3")
                shutil.move(false_path, true_path)
        except Exception as e:
            print(e)
            return None
    return (processFile,)


@app.cell
def _(STEM_FOLDER, Separator, UVR_MODEL_PATH, logging):
    UVR_MODEL_NAME = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    OUTPUT_FORMAT = "MP3"

    separator = Separator(
        model_file_dir=UVR_MODEL_PATH,
        output_dir=STEM_FOLDER,
        output_format=OUTPUT_FORMAT,
        log_level=logging.WARNING,
    )

    separator.load_model(model_filename=UVR_MODEL_NAME)
    return (separator,)


@app.cell
def _(fma_single_file, mo, processFile, triplet_df_single):
    fma_single_file["vocal_path"] = [
        processFile(int(track_id))
        for track_id in mo.status.progress_bar(
            triplet_df_single.index,
            title="Separating vocals...",
            subtitle="Processing tracks",
            completion_title="Done!",
            completion_subtitle=f"Processed {len(triplet_df_single)} tracks",
        )
    ]

    fma_single_file.dropna(subset=["vocal_path"])
    return


if __name__ == "__main__":
    app.run()
