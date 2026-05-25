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
    import pandas as pd
    import numpy as np
    import os
    import seaborn as sns

    from src.dataset.triplet_dataset import TripletDataset
    from src.globals import (
        CSV_FOLDER,
        DATASET_FOLDER,
        TRACKS_PATH,
        AUDIO_FOLDER,
        STEMS_FOLDER,
        UVR_MODEL_PATH,
        MODEL_FOLDER,
    )
    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        MODEL_FOLDER,
        TripletDataset,
        mo,
        np,
        os,
        pd,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Load Dataset
    """)
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
def _(CSV_FOLDER, os, pd):
    full_dataset = pd.read_csv(
        os.path.join(CSV_FOLDER, "LargeDataset", "dataset_all_artist_tracks.csv"),
        index_col="track_id",
    )
    full_dataset
    return (full_dataset,)


@app.cell
def _(DATASET_FOLDER, os, pd):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey")


    def parse_js_date(series):
        cleaned = series.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
        return pd.to_datetime(cleaned, format="%a %b %d %Y %H:%M:%S GMT%z")


    participants = pd.read_csv(
        os.path.join(SURVEY_FOLDER, "participants.csv"), index_col="_id"
    )
    surveyQuestions = pd.read_csv(
        os.path.join(SURVEY_FOLDER, "surveyQuestions.csv"), index_col="_id"
    )
    surveyAnswers_ = pd.read_csv(
        os.path.join(SURVEY_FOLDER, "surveyAnswers.csv"), index_col="_id"
    )
    songs = pd.read_csv(os.path.join(SURVEY_FOLDER, "songs.csv"), index_col="_id")
    participants["editDate"] = parse_js_date(participants["editDate"])
    participants["createDate"] = parse_js_date(participants["createDate"])
    participants[participants.surveyCompleted]
    participants["completionTime"] = (
        participants["editDate"] - participants["createDate"]
    )
    participants["completionMinutes"] = (
        participants["completionTime"].dt.total_seconds() / 60
    )
    participants[participants.surveyCompleted]
    surveyAnswers_
    return surveyAnswers_, surveyQuestions


@app.cell
def _(full_dataset, track_df):
    track_df["artist_id"] = track_df.apply(
        lambda x: full_dataset.loc[x.name].artist_id, axis=1
    )
    track_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Link Survey Answers to TrackIDs
    """)
    return


@app.cell
def _(pd, surveyAnswers_, surveyQuestions, track_df):
    def getTrackIdsForAnswer(row):
        questionnaireID = row["questionID"]
        answerKey1 = row["answer_1"]
        answerKey2 = row["answer_2"]
        question = surveyQuestions.loc[questionnaireID]

        def getArtistID(id):
            if question["skip"]:
                return None
            return track_df.loc[id].artist_id

        return pd.Series(
            {
                "track_id_X": question["X"],
                "artist_id_X": getArtistID(question["X"]),
                "track_id_1": question[answerKey1],
                "artist_id_1": getArtistID(question[answerKey1]),
                "track_id_2": question[answerKey2],
                "artist_id_2": getArtistID(question[answerKey2]),
                "skipped": question["skip"],
            }
        )


    surveyAnswers_[
        [
            "track_id_X",
            "artist_id_X",
            "track_id_1",
            "artist_id_1",
            "track_id_2",
            "artist_id_2",
            "skipped",
        ]
    ] = surveyAnswers_.apply(getTrackIdsForAnswer, axis=1)
    surveyAnswers = surveyAnswers_[~surveyAnswers_.skipped].drop(columns="skipped")
    surveyAnswers
    return (surveyAnswers,)


@app.cell
def _(mo):
    mo.md(r"""
    # Load Embedding Model
    - MelSpectrogramEncoder from [pip](https://speechbrain.readthedocs.io/en/latest/installation.html), thanks to speechbrain
    - Encoder Model from [HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb-mel-spec), thanks to speechbrain

    Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., Subakan, C., Dawalatabad, N., Heba, A., Zhong, J., Chou, J.-C., Yeh, S.-L., Fu, S.-W., Liao, C.-F., Rastorgueva, E., Grondin, F., Aris, W., Na, H., Gao, Y., … Bengio, Y. (2021). SpeechBrain: A General-Purpose Speech Toolkit.
    """)
    return


@app.cell
def _():
    # additional imports
    import torchaudio
    import torch
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.utils.data import Dataset, DataLoader
    from speechbrain.inference.encoders import MelSpectrogramEncoder
    from sklearn.model_selection import train_test_split

    from src.utils import get_trimmed_audio
    return (
        DataLoader,
        MelSpectrogramEncoder,
        get_trimmed_audio,
        torch,
        train_test_split,
    )


@app.cell
def _(torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE
    return (DEVICE,)


@app.cell
def _(DEVICE, MelSpectrogramEncoder):
    encoder = MelSpectrogramEncoder.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb-mel-spec",
        run_opts={"device": DEVICE.type},
    )
    return (encoder,)


@app.cell
def _(mo):
    mo.md(r"""
    # Traing Data Initialization
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Get Artist Nodes (X)
    """)
    return


@app.cell
def _():
    RANDOM_STATE = 1
    # SHUFFLE_ARTIST_TRACKS = False
    TRAIN_SIZE = 0.8
    SAMPLE_RATE = 16_000
    return RANDOM_STATE, SAMPLE_RATE, TRAIN_SIZE


@app.cell
def _(SAMPLE_RATE, encoder, get_trimmed_audio, np, track_df):
    def get_embedding(song_path):
        trimmed_audio = get_trimmed_audio(song_path, sr=SAMPLE_RATE)
        return encoder.encode_waveform(trimmed_audio).cpu().numpy().squeeze()


    x = np.stack(track_df.song_path.apply(get_embedding).values)
    track_df["edge_id"] = range(len(track_df))
    return (x,)


@app.cell
def _(
    DataLoader,
    RANDOM_STATE,
    TRAIN_SIZE,
    TripletDataset,
    surveyAnswers,
    track_df,
    train_test_split,
):
    train_df, test_df = train_test_split(
        surveyAnswers, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )

    train_dataset = TripletDataset(
        triplet_df=train_df, node_df=track_df, node_id_key="edge_id"
    )
    test_dataset = TripletDataset(
        triplet_df=test_df, node_df=track_df, node_id_key="edge_id"
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader, train_loader


@app.cell
def _(mo):
    mo.md(r"""
    # Load and Train GASTSY Model
    """)
    return


@app.cell
def _():
    from src.gatsy.architectures import GATSY
    from src.gatsy.model import Trainer
    return GATSY, Trainer


@app.cell
def _(x):
    model_params = {
        "n_heads": 2,
        "n_layers": 1,
        "input_shape": x.shape[-1],
        "layer_size": 32,
    }
    return (model_params,)


@app.cell
def _(GATSY, model_params):
    model = GATSY(**model_params)
    return (model,)


@app.cell
def _(DEVICE, MODEL_FOLDER, model, os, test_loader, train_loader, x):
    trainer_params = {
        "model": model,
        "train_loader": train_loader,
        "train_x": x,
        "test_loader": test_loader,
        "lr": 0.0001,
        "epochs": 20,
        "margin": 0.5,
        "device": DEVICE,
        "n_neighbors": 5,
        "weight_decay": 0.0001,
        "model_path": os.path.join(MODEL_FOLDER, "GATSY", "escapa"),
    }
    return (trainer_params,)


@app.cell
def _(Trainer, trainer_params):
    trainer = Trainer(**trainer_params)
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell
def _(trainer):
    trainer.checkpoint
    return


@app.cell
def _(os, torch, trainer_params):
    loaded_checkpoint = torch.load(
        os.path.join(
            trainer_params["model_path"], "gatsy_30_0.0001_20260523-185154.pt"
        )
    )
    loaded_checkpoint.keys()
    return


if __name__ == "__main__":
    app.run()
