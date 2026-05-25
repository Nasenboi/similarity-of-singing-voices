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
    import matplotlib.pyplot as plt

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
        plt,
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
    # Load Opensmile Utilities
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
    import opensmile

    from src.utils import get_trimmed_audio
    return DataLoader, opensmile, torch, train_test_split


@app.cell
def _(torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE
    return (DEVICE,)


@app.cell
def _(opensmile):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    smile.feature_names
    return


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
    return RANDOM_STATE, TRAIN_SIZE


@app.cell
def _(DATASET_FOLDER, np, os, track_df):
    """
    def get_feature_set(song_path):
        trimmed_audio = get_trimmed_audio(song_path, sr=SAMPLE_RATE)
        return smile.process_signal(trimmed_audio, SAMPLE_RATE).values[0]


    x = np.stack(track_df.song_path.apply(get_feature_set).values)
    """

    gemaps_feature_path = os.path.join(
        DATASET_FOLDER, "fma_large_feature_sets", "gemaps.npy"
    )
    x = np.load(gemaps_feature_path)
    track_df["edge_id"] = range(len(track_df))
    x
    return (x,)


@app.cell
def _(x):
    x.shape
    return


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
def _(mo):
    mo.md(r"""
    ## Pretrained Model Parameter Fetching
    """)
    return


@app.cell
def _():
    from src.gatsy.architectures import GATSY
    from src.gatsy.model import Trainer
    return GATSY, Trainer


@app.cell
def _(GATSY):
    import sys
    from unittest.mock import MagicMock

    mock_module = MagicMock()
    mock_module.GATSY = GATSY

    sys.modules["src"] = MagicMock()
    sys.modules["src.architectures"] = mock_module
    return


@app.cell
def _(MODEL_FOLDER, os, torch):
    pt_model_path = os.path.join(
        MODEL_FOLDER, "GATSY", "pretrained", "GAT_7_3_1_0.001_0.0_triplet_False.pt"
    )

    pt_checkpoint = torch.load(pt_model_path, weights_only=False)
    state_dict = pt_checkpoint.state_dict()
    for l in [
        "gat_layers.0.att_src",
        "gat_layers.0.att_dst",
        "gat_layers.0.bias",
        "gat_layers.0.lin.weight",
        "linear2.weight",
        "linear2.bias",
    ]:
        del state_dict[l]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Initialization
    """)
    return


@app.cell
def _(GATSY, x):
    model_params = {
        "n_heads": 1,
        "n_layers": 2,
        "input_dim": x.shape[-1],
        "hidden_dim": 256,
        "output_dim": 128,
    }
    model = GATSY(**model_params)
    return (model,)


@app.cell
def _():
    """
    model.load_state_dict(state_dict, strict=False)

    freeze_layers = [
        "gat_layers.0.lin.weight",
        "gat_layers.1.lin.weight",
        "gat_layers.2.att_src",
        "gat_layers.2.att_dst",
        "gat_layers.2.bias",
        "gat_layers.2.lin.weight",
    ]
    for name, param in model.named_parameters():
        if any(layer in name for layer in freeze_layers):
            param.requires_grad = False
    """
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Setting up and Running Trainer
    """)
    return


@app.cell
def _(
    DEVICE,
    MODEL_FOLDER,
    Trainer,
    model,
    os,
    plt,
    test_loader,
    train_loader,
    x,
):
    trainer_params = {
        "model": model,
        "train_loader": train_loader,
        "train_x": x,
        "test_loader": test_loader,
        "lr": 0.0001,
        "epochs": 250,
        "early_stop": 10,
        "margin": 1.5,
        "device": DEVICE,
        "n_neighbors": 10,
        "weight_decay": 0.0001,
        "model_path": os.path.join(MODEL_FOLDER, "GATSY", "gemaps"),
    }
    trainer = Trainer(**trainer_params)
    trainer.train()
    loss_reduction = -1 * (
        trainer.best_test_loss - trainer.checkpoint["loss_test"][0]
    )
    loss_recuction_perc = 100 * loss_reduction / trainer.checkpoint["loss_test"][0]

    print(f"Loss reduction {loss_reduction} ({loss_recuction_perc:.2f}%)")
    plt.plot(trainer.checkpoint["loss_test"], label="Test Loss")
    plt.plot(trainer.checkpoint["loss_train"], label="Train Loss", color="orange")
    plt.legend()
    plt.show()
    return (trainer,)


@app.cell
def _(plt, trainer):
    plt.plot(trainer.checkpoint["accuracy_score"])
    return


@app.cell
def _(plt, trainer):
    plt.plot(trainer.checkpoint["pred_accuracy"])
    return


if __name__ == "__main__":
    app.run()
