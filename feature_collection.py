import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import marimo as mo
    import numpy as np
    import pandas as pd
    import os

    from src.globals import CSV_FOLDER
    from src.utils import get_trimmed_audio

    from src.FMA.utils import load, get_audio_path
    from src.globals import (
        CSV_FOLDER,
        TRACKS_PATH,
        AUDIO_FOLDER,
        STEMS_FOLDER,
        UVR_MODEL_PATH,
        DATASET_FOLDER,
        MODEL_FOLDER,
    )
    return CSV_FOLDER, DATASET_FOLDER, MODEL_FOLDER, mo, os, pd


@app.cell
def _(DATASET_FOLDER, os, pd):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey")
    songs = pd.read_csv(
        os.path.join(SURVEY_FOLDER, "songs.csv"), index_col="trackID"
    )
    songs
    return (songs,)


@app.cell
def _(mo):
    mo.md(r"""
    # Load the Tracks DF
    Use only songs that were not skipped in the survey.
    Drop columns with unimportant values:
    column | reason
    checked => always true
    is voiced  => always true
    interview => always false
    artist overlaps  => always empty list
    multiple voices  => always false
    voice quality  => always 3
    """)
    return


@app.cell
def _(CSV_FOLDER, os, pd, songs):
    track_df = pd.read_csv(
        os.path.join(
            CSV_FOLDER,
            "LargeDataset",
            "triplet_selection",
            "dataset_vq3_finished.csv",
        ),
        index_col="track_id",
    ).join(songs["skipInSurvey"])
    track_df = track_df[~track_df["skipInSurvey"]].drop(
        columns=[
            "skipInSurvey",
            "checked",
            "is_voiced",
            "interview",
            "artist_overlaps",
            "multiple_voices",
            "voice_quality",
        ]
    )
    track_df
    return (track_df,)


@app.cell
def _(mo):
    mo.md(r"""
    # Get Additional Features

    Most of them will be estimated by machine learning techniques

    ## Song Information
    (by essentia)
    - genre (Discogs400)
    - Approachability
    - Engagement
    - ArousalValence MuSe
    - Dancability
    - Mood MIREX
    - tempo (deeptemp-k16)

    ## High Level Singer Information
    - age
    - gender
    - accent
    - language
    - timbre
    -
    """)
    return


@app.cell
def _(MODEL_FOLDER, os):
    import librosa as lr
    import essentia as es
    from essentia import Pool
    from essentia.standard import (
        MonoLoader,
        TensorflowPredictMAEST,
        TensorflowPredict,
    )

    # https://essentia.upf.edu/models.html#genre-discogs400
    # embeddings:
    maest30sFilename = os.path.join(
        MODEL_FOLDER, "genre_discogs400", "discogs-maest-30s-pw-2.pb"
    )
    # classification:
    predict30sFilename = os.path.join(
        MODEL_FOLDER,
        "genre_discogs400",
        "genre_discogs400-discogs-maest-30s-pw-1.pb",
    )

    embedding_model = TensorflowPredictMAEST(
        graphFilename=maest30sFilename, output="PartitionedCall/Identity_12"
    )
    classifier_model = TensorflowPredict(
        graphFilename=predict30sFilename,
        inputs=["embeddings"],
        outputs=["PartitionedCall/Identity_1"],
    )
    return MonoLoader, Pool, classifier_model, embedding_model


@app.cell
def _(MonoLoader, Pool, classifier_model, embedding_model, track_df):
    def get_genre_predictions(path: str):
        audio = MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()

        embeddings = embedding_model(audio)

        pool = Pool()
        pool.set("embeddings", embeddings)
        predictions = classifier_model(pool)["PartitionedCall/Identity_1"]
        print(predictions)

        return predictions.mean(axis=0)


    get_genre_predictions(track_df.iloc[0].song_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
