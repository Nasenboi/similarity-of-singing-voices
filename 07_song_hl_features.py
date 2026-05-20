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
    import marimo as mo
    import marimo as mo
    import numpy as np
    import pandas as pd
    import os

    from src.globals import CSV_FOLDER

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
    return CSV_FOLDER, DATASET_FOLDER, MODEL_FOLDER, mo, np, os, pd


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
    Drop columns with irrelevant values:
    - checked => always true
    - is voiced  => always true
    - interview => always false
    - artist overlaps  => always empty list
    - multiple voices  => always false
    - voice quality  => always 3
    """)
    return


@app.cell
def _(CSV_FOLDER, os, pd, songs):
    track_df = pd.read_csv(
        os.path.join(
            CSV_FOLDER,
            "LargeDataset",
            "additional_features",
            "genres.csv",
        ),
        index_col="track_id",
    ).join(songs["skipInSurvey"])
    track_df = track_df[~track_df["skipInSurvey"]]
    """    
    .drop(
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
    """
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
def _():
    """
    Initial Test Code and Genre Prediction 

    # https://essentia.upf.edu/models.html#genre-discogs400
    # embeddings:
    embeddingFilename = os.path.join(
        MODEL_FOLDER, "effnet", "discogs-effnet-bs64-1.pb"
    )
    # classification:
    predictFilename = os.path.join(
        MODEL_FOLDER,
        "genre_discogs400",
        "genre_discogs400-discogs-effnet-1.pb",
    )
    metadataFilename = os.path.join(
        MODEL_FOLDER,
        "genre_discogs400",
        "genre_discogs400-discogs-effnet-1.json"
    )

    with open(metadataFilename) as json_file:
        metadata = json.load(json_file)

    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=embeddingFilename, output="PartitionedCall:1")
    model = TensorflowPredict2D(graphFilename=predictFilename, input="serving_default_model_Placeholder", output="PartitionedCall:0")

    def get_genre_predictions(path: str) -> dict:
        real_path = path.replace(os.getenv("DRIVE"), "/data")
        audio = MonoLoader(filename=real_path, sampleRate=16000, resampleQuality=4)()
        embeddings = embedding_model(audio)

        predictions = model(embeddings).mean(axis=0)

        genreString = metadata["classes"][predictions.argmax()]

        genreMain, genreSub =  genreString.split("---")

        return {
            "pred_genre_string": genreString,
            "pred_genre_main": genreMain,
            "pred_genre_sub": genreSub,
        }

    track_df[["pred_genre_string", "pred_genre_main", "pred_genre_sub"]] = track_df["song_path"].apply(
        lambda p: pd.Series(get_genre_predictions(p))
    )
    track_df
    """
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Initialize Essentia ML Embeddings
    - Embeddings done via [Effnet Discogs](https://essentia.upf.edu/models.html#discogs-effnet)

    Alonso-Jiménez, P., Serra, X., & Bogdanov, D. (2023). Efficient Supervised Training of Audio Transformers for Music Representation Learning. https://arxiv.org/abs/2309.16418
    """)
    return


@app.cell
def _(os):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import librosa as lr
    import essentia as es
    from essentia import Pool
    from essentia.standard import (
        MonoLoader,
        TensorflowPredictEffnetDiscogs,
        TensorflowPredict2D,
    TempoCNN
    )
    import tensorflow as tf
    import json

    tf.debugging.set_log_device_placement(True)
    print(tf.config.list_physical_devices('GPU'))
    return (
        MonoLoader,
        TempoCNN,
        TensorflowPredict2D,
        TensorflowPredictEffnetDiscogs,
        json,
        tf,
    )


@app.cell
def _(MODEL_FOLDER, TensorflowPredictEffnetDiscogs, os, tf):
    # generate embeddings
    embeddingFilename = os.path.join(
        MODEL_FOLDER, "effnet", "discogs-effnet-bs64-1.pb"
    )
    with tf.device('/GPU:0'):
        embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=embeddingFilename, output="PartitionedCall:1")
    return (embedding_model,)


@app.cell
def _(MonoLoader, embedding_model, mo, os, pd, track_df):
    def getSongEmbeddings(path: str):
        real_path = path.replace(os.getenv("DRIVE"), "/data")
        audio = MonoLoader(filename=real_path, sampleRate=16000, resampleQuality=4)()
        return embedding_model(audio)

    song_paths = track_df["song_path"].tolist()
    embeddings_list = []

    for path in mo.status.progress_bar(
        song_paths,
        title="Generating Embeddings",
        subtitle="Processing songs...",
        completion_title="Embeddings Complete",
        completion_subtitle=f"Processed {len(song_paths)} songs",
        show_rate=True,
        show_eta=True,
    ):
        embeddings_list.append(getSongEmbeddings(path))

    songEmbeddings = pd.Series(embeddings_list, index=track_df.index)
    songEmbeddings
    return (embeddings_list,)


@app.cell
def _(MODEL_FOLDER, TensorflowPredict2D, embeddings_list, mo, os, track_df):
    # approachability
    approachabilityModelName = os.path.join(
        MODEL_FOLDER, "approachability", "approachability_regression-discogs-effnet-1.pb")
    approachabilityModel = TensorflowPredict2D(graphFilename=approachabilityModelName, output="model/Identity")

    track_df["pred_approachability"] = [
        approachabilityModel(emb).mean(axis=0)
        for emb in mo.status.progress_bar(
            embeddings_list,
            title="Predicting Approachability",
            show_rate=True,
            show_eta=True,
        )
    ]
    del approachabilityModelName, approachabilityModel
    track_df
    return


@app.cell
def _(model, modelName):
    del modelName, model
    return


@app.cell
def _(
    MODEL_FOLDER,
    TensorflowPredict2D,
    embeddings_list,
    mo,
    np,
    os,
    track_df,
):
    danceModelName = os.path.join(MODEL_FOLDER, "danceability", "danceability-discogs-effnet-1.pb")
    danceModel = TensorflowPredict2D(graphFilename=danceModelName, output="model/Softmax")

    track_df[["pred_danceable", "pred_not_danceable"]] = np.vstack([
        danceModel(emb).mean(axis=0)
        for emb in mo.status.progress_bar(
            embeddings_list,
            title="Predicting Danceability",
            show_rate=True,
            show_eta=True,
        )
    ])

    del danceModelName, danceModel
    track_df
    return


@app.cell
def _(MODEL_FOLDER, TensorflowPredict2D, embeddings_list, mo, os, track_df):
    # engageability
    engageModelName = os.path.join(
        MODEL_FOLDER, "engagement", "engagement_regression-discogs-effnet-1.pb")
    engageModel = TensorflowPredict2D(graphFilename=engageModelName, output="model/Identity")

    track_df["pred_engagement"] = [
        engageModel(emb).mean(axis=0)
        for emb in mo.status.progress_bar(
            embeddings_list,
            title="Predicting Engagement",
            show_rate=True,
            show_eta=True,
        )
    ]

    del engageModelName, engageModel
    track_df
    return


@app.cell
def _(
    MODEL_FOLDER,
    TensorflowPredict2D,
    embeddings_list,
    json,
    mo,
    np,
    os,
    track_df,
):
    moodModelName = os.path.join(MODEL_FOLDER, "moodAndTheme", "mtg_jamendo_moodtheme-discogs-effnet-1.pb")
    moodModel = TensorflowPredict2D(graphFilename=moodModelName)
    moodModelMedatadaPath = os.path.join(MODEL_FOLDER, "moodAndTheme", "mtg_jamendo_moodtheme-discogs-effnet-1.json")

    with open(moodModelMedatadaPath) as json_file:
        moodModelClasses = json.load(json_file)["classes"]

    track_df[["pred_mood_and_theme"]] = np.vstack([
        moodModelClasses[moodModel(emb).mean(axis=0).argmax()]
        for emb in mo.status.progress_bar(
            embeddings_list,
            title="Predicting Mood and Theme",
            show_rate=True,
            show_eta=True,
        )
    ])

    del moodModelName, moodModel, moodModelMedatadaPath, moodModelClasses
    track_df
    return


@app.cell
def _(MODEL_FOLDER, MonoLoader, TempoCNN, mo, os, track_df):
    tempoModelName = os.path.join(MODEL_FOLDER, "tempo", "deeptemp-k4-3.pb")
    tempoModel = TempoCNN(graphFilename=tempoModelName)

    def gettempo(path: str):
        real_path = path.replace(os.getenv("DRIVE"), "/data")
        audio = MonoLoader(filename=real_path, sampleRate=11025, resampleQuality=4)()
        global_tempo, local_tempo, local_tempo_probabilities = tempoModel(audio)
        return global_tempo  # single BPM float

    track_df["pred_tempo"] = [
        gettempo(path)
        for path in mo.status.progress_bar(
            track_df["song_path"].tolist(),
            title="Predicting Tempo",
            show_rate=True,
            show_eta=True,
        )
    ]

    del tempoModelName, tempoModel
    track_df
    return


@app.cell
def _(DATASET_FOLDER, embeddings_list, np, os):
    embeddingsPath = os.path.join(DATASET_FOLDER, "fma_large_embeddings", "discogs-effnet.npy")
    np.save(embeddingsPath, np.array(embeddings_list, dtype=object), allow_pickle=True)
    return


if __name__ == "__main__":
    app.run()
