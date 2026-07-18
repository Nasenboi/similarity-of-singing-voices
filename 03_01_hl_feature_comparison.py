import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    def # Import Python Packages
    """)
    return


@app.cell
def _():
    import os
    from typing import List, Literal

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import torchaudio
    from sklearn.preprocessing import StandardScaler
    from speechbrain.inference.encoders import MelSpectrogramEncoder

    from src import load_singer_identity_model
    from src.globals import (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        MODEL_FOLDER,
        PLOT_FOLDER,
        STEMS_FOLDER,
        TRACKS_PATH,
        UVR_MODEL_PATH,
    )
    from src.statistics.feature_correlation import (
        get_all_distance_differences,
        get_global_distance_scores,
        scale_df,
    )
    from src.statistics.plotting import (
        plot_correlation_bar,
        plot_correlation_scatter,
    )
    from src.survey_dataset_helpers import load_survey_data
    from src.utils import get_trimmed_audio

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        PLOT_FOLDER,
        get_all_distance_differences,
        load_survey_data,
        mo,
        os,
        plot_correlation_bar,
        plot_correlation_scatter,
        plt,
        scale_df,
    )


@app.cell
def _(CSV_FOLDER, DATASET_FOLDER, os):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey", "survey_2")
    CSV_PATHS = {
        "participants": os.path.join(SURVEY_FOLDER, "participants.csv"),
        "songs": os.path.join(SURVEY_FOLDER, "songs.csv"),
        "answers": os.path.join(SURVEY_FOLDER, "surveyAnswers.csv"),
        "questions": os.path.join(SURVEY_FOLDER, "surveyQuestions.csv"),
        "tracks": os.path.join(
            CSV_FOLDER,
            "LargeDataset",
            "dataset_survey_2_final.csv",
        ),
    }
    return (CSV_PATHS,)


@app.cell
def _(PLOT_FOLDER, os):
    PLOT_SAVE_DIR = os.path.join(PLOT_FOLDER, "survey_2")
    return (PLOT_SAVE_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Load Dataset
    """)
    return


@app.cell
def _(CSV_PATHS, load_survey_data):
    SURVEY_DATA = load_survey_data(CSV_PATHS)
    questions_df = SURVEY_DATA["questions_df"]
    answers_df = SURVEY_DATA["answers_df"]
    participants_df = SURVEY_DATA["participants_df"]
    songs_df = SURVEY_DATA["songs_df"]
    human_agreement = SURVEY_DATA["human_agreement"]
    answer_a_b_ratio = SURVEY_DATA["answer_a_b_ratio"]
    track_df = SURVEY_DATA["track_df"]
    return questions_df, track_df


@app.cell
def _(PLOT_FOLDER, os, plt, track_df):
    fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(14, 7))

    genre_counts_1 = track_df["genre_top"].value_counts()
    ax11.pie(
        genre_counts_1.values,
        labels=[
            f"{label}: {count}" for label, count in genre_counts_1.items()
        ],
        autopct="%1.1f%%",
        startangle=5,
    )
    ax11.set_title("Genre", fontsize=12, pad=10)

    gender_counts_1 = track_df["pred_gender"].value_counts()
    ax12.pie(
        gender_counts_1.values,
        labels=[
            f"{label}: {count}" for label, count in gender_counts_1.items()
        ],
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
        os.path.join(PLOT_FOLDER, "survey_2", "feature_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Single High Level Feature Agreement Scores
    """)
    return


@app.cell
def _():
    hl_features = [
        "pred_genre_main",
        "pred_genre_sub",
        "pred_approachability",
        "pred_danceable",
        "pred_not_danceable",
        "pred_engagement",
        "pred_mood_and_theme",
        "pred_tempo",
        "pred_gender",
        "pred_p_male",
        "pred_p_female",
        "pred_age",
        "pred_age_no_trim",
    ]
    return (hl_features,)


@app.cell
def _(
    get_all_distance_differences,
    hl_features,
    questions_df,
    scale_df,
    track_df,
):
    scaled_track_df = scale_df(
        track_df,
        [
            "pred_approachability",
            "pred_danceable",
            "pred_not_danceable",
            "pred_engagement",
            "pred_tempo",
            "pred_p_male",
            "pred_p_female",
            "pred_age",
            "pred_age_no_trim",
        ],
    )
    hl_distances = get_all_distance_differences(
        scaled_track_df, hl_features, questions_df
    )
    hl_distances
    return (hl_distances,)


@app.cell
def _(hl_distances, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="High Level Feature Correlations",
        feature_df=hl_distances,
        target_feature=questions_df["A_perc"],
        top_x=len(hl_distances.columns),
    )
    return


@app.cell
def _(hl_distances, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="High Level Feature Correlations (Randomized)",
        feature_df=hl_distances[questions_df.randomized],
        target_feature=questions_df[questions_df.randomized]["A_perc"],
        top_x=len(hl_distances.columns),
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, hl_distances, plot_correlation_scatter, questions_df):
    plot_correlation_scatter(feature_name="Gender", x=questions_df["A_perc"], y=hl_distances["pred_gender"], plot_dir=PLOT_SAVE_DIR)
    return


if __name__ == "__main__":
    app.run()
