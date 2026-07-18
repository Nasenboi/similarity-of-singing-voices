import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Import Python Packages
    """)
    return


@app.cell
def _():
    import os
    from typing import List, Literal

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import opensmile
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
        get_distance_row,
        get_global_distance_scores,
        scale_df,
    )
    from src.statistics.feature_correlation import get_anova_values
    from src.statistics.plotting import (
        plot_correlation_bar,
        plot_correlation_scatter,
    )
    from src.survey_dataset_helpers import load_survey_data, get_answer_ratios
    from src.utils import get_trimmed_audio

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        get_all_distance_differences,
        get_anova_values,
        get_answer_ratios,
        load_survey_data,
        mo,
        os,
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
    return answers_df, questions_df, track_df


@app.cell
def _(get_all_distance_differences, questions_df, scale_df, track_df):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run ANOVA

    ## Randomized vs Max Entropy
    """)
    return


@app.cell
def _(get_answer_ratios, questions_df):
    def get_ageement_values(answers_df):
        return questions_df.apply(
            lambda x: get_answer_ratios(x.name, answers_df).agreement, axis=1
        ).dropna(axis=0)

    return (get_ageement_values,)


@app.cell
def _(answers_df, questions_df):
    randomized_mask = answers_df.questionID.apply(
        lambda x: questions_df.loc[x].randomized
    )
    return (randomized_mask,)


@app.cell
def _(answers_df, get_ageement_values, randomized_mask):
    (
        get_ageement_values(answers_df[randomized_mask]).mean(),
        get_ageement_values(answers_df[~randomized_mask]).mean(),
    )
    return


@app.cell
def _(answers_df, get_ageement_values, get_anova_values, randomized_mask):
    get_anova_values(
        get_ageement_values(answers_df[randomized_mask]),
        get_ageement_values(answers_df[~randomized_mask]),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gender Distribution
    """)
    return


@app.cell
def _(answers_df, questions_df):
    gender_distribution_mask = answers_df.questionID.apply(
        lambda x: questions_df.loc[x].gender_distribution == 0.5
    )
    return (gender_distribution_mask,)


@app.cell
def _(answers_df, gender_distribution_mask, get_ageement_values):
    (
        get_ageement_values(answers_df[gender_distribution_mask]).mean(),
        get_ageement_values(answers_df[~gender_distribution_mask]).mean(),
    )
    return


@app.cell
def _(
    answers_df,
    gender_distribution_mask,
    get_ageement_values,
    get_anova_values,
):
    get_anova_values(
        get_ageement_values(answers_df[gender_distribution_mask]),
        get_ageement_values(answers_df[~gender_distribution_mask]),
    )
    return


if __name__ == "__main__":
    app.run()
