import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import Python Packages
    """)
    return


@app.cell
def _():
    import os
    import pathlib
    import altair as alt
    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    from src.globals import (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        MODEL_FOLDER,
        STEMS_FOLDER,
        TRACKS_PATH,
        UVR_MODEL_PATH,
        PLOT_FOLDER,
    )
    from src.phoneme_extractor.phoneme_extractor import (
        load_data as load_phoneme_data,
    )
    from src.plotting import plot_scores

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        PLOT_FOLDER,
        load_phoneme_data,
        mo,
        os,
        pd,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Dataset
    """)
    return


@app.cell
def _(DATASET_FOLDER, PLOT_FOLDER, os, torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phoneme_save_path = os.path.join(DATASET_FOLDER, "fma_large_phonemes")
    SAMPLE_RATE = 16_000
    PLOT_SAVE_DIR = os.path.join(PLOT_FOLDER, "survey_2")
    return PLOT_SAVE_DIR, phoneme_save_path


@app.cell
def _():
    MIN_PHONEME_DURATION_MS = 40
    MIN_PHONEME_CONFIDENCE = 0
    return MIN_PHONEME_CONFIDENCE, MIN_PHONEME_DURATION_MS


@app.cell
def _(
    MIN_PHONEME_CONFIDENCE,
    MIN_PHONEME_DURATION_MS,
    load_phoneme_data,
    phoneme_save_path,
):
    phoneme_df, phonemes = load_phoneme_data(phoneme_save_path)
    phoneme_df = phoneme_df.rename(columns={"file_id": "track_id"})

    phoneme_mask = (phoneme_df.duration_ms >= MIN_PHONEME_DURATION_MS) & (
        phoneme_df.confidence >= MIN_PHONEME_CONFIDENCE
    )
    phonemes = phonemes[phoneme_mask]
    phoneme_df = phoneme_df[phoneme_mask]
    phoneme_df
    return (phoneme_df,)


@app.cell
def _(phoneme_df):
    phoneme_df.groupby("track_id").size().agg(["min", "mean"])
    return


@app.cell
def _(CSV_FOLDER, os, pd, phoneme_df):
    track_df = pd.read_csv(
        os.path.join(
            CSV_FOLDER,
            "LargeDataset",
            "additional_features",
            "high_level_features.csv",
        ),
        index_col="track_id",
    )
    track_df = track_df[track_df.index.isin(phoneme_df.track_id)]
    track_df["language"] = phoneme_df.groupby("track_id")["language"].first()
    track_df["phoneme_confidence"] = (
        phoneme_df.groupby("track_id")["confidence"].sum()
        / phoneme_df.groupby("track_id")["confidence"].count()
    )
    track_df["creation_date"] = pd.to_datetime(track_df["creation_date"])
    track_df["release_date"] = pd.to_datetime(track_df["release_date"])
    track_df
    return (track_df,)


@app.cell
def _(DATASET_FOLDER, os, pd):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey", "survey_1")


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
def _(pd, phoneme_df, surveyAnswers_, surveyQuestions):
    def getTrackIdsForAnswer(row):
        questionnaireID = row["questionID"]
        answerKey1 = row["answer_1"]
        answerKey2 = row["answer_2"]
        question = surveyQuestions.loc[questionnaireID]
        return pd.Series(
            {
                "track_id_X": question["X"],
                "track_id_1": question[answerKey1],
                "track_id_2": question[answerKey2],
                "skipped": question["skip"],
            }
        )


    surveyAnswers_[["track_id_X", "track_id_1", "track_id_2", "skipped"]] = (
        surveyAnswers_.apply(getTrackIdsForAnswer, axis=1)
    )
    survey_answers_mask = (
        (~surveyAnswers_.skipped)
        & surveyAnswers_.track_id_X.isin(phoneme_df.track_id)
        & surveyAnswers_.track_id_1.isin(phoneme_df.track_id)
        & surveyAnswers_.track_id_2.isin(phoneme_df.track_id)
    )
    surveyAnswers = surveyAnswers_[survey_answers_mask].drop(columns="skipped")
    surveyAnswers
    return (surveyAnswers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset Cut
    """)
    return


@app.cell
def _(pd, track_df):
    MAX_TRACKS_PER_GENRE = 18

    content_length_mask = track_df.vocal_content_length_s >= 10
    release_mask = track_df.release_date >= pd.Timestamp("2003-01-01")
    language_mask = track_df.language == "en"
    phoneme_confidence_mask = track_df.phoneme_confidence >= 0.05
    mask = (
        language_mask
        & content_length_mask
        & release_mask
        & phoneme_confidence_mask
    )
    dropna_columns = ["genre_top", "release_date"]
    cut_track_df = track_df[mask].dropna(subset=dropna_columns)

    cut_track_df = (
        cut_track_df.sort_values("phoneme_confidence", ascending=False)
        .groupby("genre_top")
        .head(MAX_TRACKS_PER_GENRE)
    )

    cut_track_df = (
        cut_track_df.groupby("pred_gender").head(
            cut_track_df.groupby("pred_gender")["pred_gender"].count().min()
        )
    ).sort_index()

    cut_track_df
    return (cut_track_df,)


@app.cell
def _(PLOT_FOLDER, os, plt, track_df):
    fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(14, 7))

    genre_counts_1 = track_df["genre_top"].value_counts()
    ax11.pie(
        genre_counts_1.values,
        labels=[f"{label}: {count}" for label, count in genre_counts_1.items()],
        autopct="%1.1f%%",
        startangle=5,
    )
    ax11.set_title("Genre", fontsize=12, pad=10)

    gender_counts_1 = track_df["pred_gender"].value_counts()
    ax12.pie(
        gender_counts_1.values,
        labels=[f"{label}: {count}" for label, count in gender_counts_1.items()],
        autopct="%1.1f%%",
        startangle=0,
    )
    ax12.set_title("Gender", fontsize=12, pad=10)

    fig1.suptitle(
        f"Dataset Feature Distributions (Tracks: {len(track_df)})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_FOLDER, "survey_1", "feature_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    return


@app.cell
def _(PLOT_SAVE_DIR, cut_track_df, os, plt):
    fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(14, 7))

    genre_counts_2 = cut_track_df["genre_top"].value_counts()
    ax21.pie(
        genre_counts_2.values,
        labels=[f"{label}: {count}" for label, count in genre_counts_2.items()],
        autopct="%1.1f%%",
        startangle=0,
    )
    ax21.set_title("Genre", fontsize=12, pad=10)

    gender_counts_2 = cut_track_df["pred_gender"].value_counts()
    ax22.pie(
        gender_counts_2.values,
        labels=[f"{label}: {count}" for label, count in gender_counts_2.items()],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax22.set_title("Gender", fontsize=12, pad=10)

    fig2.suptitle(
        f"Dataset Feature Distributions (Tracks: {len(cut_track_df)})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_SAVE_DIR, "feature_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    return


@app.cell
def _(cut_track_df, surveyAnswers):
    surveyAnswers[
        surveyAnswers["track_id_X"].isin(cut_track_df.index)
        & surveyAnswers["track_id_1"].isin(cut_track_df.index)
        & surveyAnswers["track_id_2"].isin(cut_track_df.index)
    ]
    return


if __name__ == "__main__":
    app.run()
