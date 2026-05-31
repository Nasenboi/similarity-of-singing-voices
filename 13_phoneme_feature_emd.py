import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


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
    )
    from src.phoneme_extractor.phoneme_extractor import (
        load_data as load_phoneme_data,
    )

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        librosa,
        load_phoneme_data,
        mo,
        np,
        os,
        pd,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Dataset
    """)
    return


@app.cell
def _(DATASET_FOLDER, os, torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phoneme_save_path = os.path.join(DATASET_FOLDER, "fma_large_phonemes")
    SAMPLE_RATE = 16_000
    return SAMPLE_RATE, phoneme_save_path


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
    return phoneme_df, phonemes


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
    track_df
    return


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
    # Reusable Functions
    """)
    return


@app.cell
def _():
    from scipy.stats import wasserstein_distance_nd

    return (wasserstein_distance_nd,)


@app.cell
def _(mo, pd, phoneme_df, surveyAnswers, wasserstein_distance_nd):
    def get_emd_difference(x, a_1, a_2) -> dict:
        """Get the differences of EMD-Distances for a triplet of distributions
        x, a_1 and a_2 should be a distribution as nd-vectors of feature values
        """
        emd_1 = wasserstein_distance_nd(x, a_1)
        emd_2 = wasserstein_distance_nd(x, a_2)
        return (emd_2 - emd_1) / (emd_1 + emd_2)


    def get_phoneme_features_per_track(track_id, feature_df):
        return feature_df[phoneme_df.track_id == track_id]


    def get_local_feature_emd_distance(
        answer,
        feature_df: pd.DataFrame,
    ):
        x = get_phoneme_features_per_track(answer["track_id_X"], feature_df)
        a_1 = get_phoneme_features_per_track(answer["track_id_1"], feature_df)
        a_2 = get_phoneme_features_per_track(answer["track_id_2"], feature_df)
        return get_emd_difference(x, a_1, a_2)


    def run_emd_distance_algorithm(feature_df):
        rows = mo.status.progress_bar(
            surveyAnswers.iterrows(),
            title="Calculating EMD Distance Algorithm",
            total=len(surveyAnswers),
            remove_on_exit=True,
        )
        return pd.Series(
            {
                idx: get_local_feature_emd_distance(row, feature_df)
                for idx, row in rows
            }
        )

    return (run_emd_distance_algorithm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MFCCs
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler

    return (StandardScaler,)


@app.cell
def _(SAMPLE_RATE):
    mfcc_config = {
        "n_mfcc": 10,
        "hop_length": int(SAMPLE_RATE * 0.01),
        "fmin": 50,
        "fmax": 8_000,
    }
    return (mfcc_config,)


@app.cell
def _(SAMPLE_RATE, librosa):
    def get_mfcc_stats(
        y, n_mfcc=10, hop_length=int(SAMPLE_RATE * 0.01), fmin=50, fmax=8_000
    ):
        """Calculates mel-frequency cepstral coefficients, sampled at SAMPLE_RATE (16 kHz)"""
        n_fft = min(2048, len(y))
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=SAMPLE_RATE,
            n_mfcc=n_mfcc,
            fmin=fmin,
            fmax=fmax,
            hop_length=hop_length,
            n_fft=n_fft,
        )
        return mfccs.mean(axis=1)

    return (get_mfcc_stats,)


@app.cell
def _(
    StandardScaler,
    get_mfcc_stats,
    mfcc_config,
    mo,
    np,
    pd,
    phoneme_df,
    phonemes,
):
    mfcc_scaler = StandardScaler()

    mfcc_values = np.stack(
        [
            get_mfcc_stats(x, **mfcc_config)
            for x in mo.status.progress_bar(
                phonemes, title="Calculating MFCCs...", remove_on_exit=True
            )
        ]
    )

    mfcc_df = pd.DataFrame(
        mfcc_scaler.fit_transform(mfcc_values),
        columns=[f"mfcc_{e}" for e in range(mfcc_config["n_mfcc"])],
        index=phoneme_df.index,
    )
    mfcc_df
    return (mfcc_df,)


@app.cell
def _(mfcc_df, run_emd_distance_algorithm):
    mfcc_emd_differences = run_emd_distance_algorithm(mfcc_df)
    mfcc_emd_differences
    return (mfcc_emd_differences,)


@app.cell
def _(mfcc_emd_differences):
    mfcc_emd_differences.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mel Frequencies
    """)
    return


@app.cell
def _():
    mel_config = {
        "fmin": 50,
        "fmax": 8_000,
        "n_mels": 128,
    }
    return (mel_config,)


@app.cell
def _(SAMPLE_RATE, librosa):
    def get_mel_frequencies(y, fmin=50, fmax=8_000, n_mels=128):
        """Calculates mel-frequency energy values from audio samples (y), sampled at SAMPLE_RATE (16 kHz)"""
        n_fft = min(2048, len(y))
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=n_mels, fmin=fmin, fmax=fmax, n_fft=n_fft
        )
        return librosa.power_to_db(mel_spec).mean(axis=1)  # shape: (n_mels,)

    return (get_mel_frequencies,)


@app.cell
def _(
    StandardScaler,
    get_mel_frequencies,
    mel_config,
    mo,
    np,
    pd,
    phoneme_df,
    phonemes,
):
    mel_scaler = StandardScaler()

    mel_values = np.stack(
        [
            get_mel_frequencies(x, **mel_config)
            for x in mo.status.progress_bar(
                phonemes, title="Calculating Mels...", remove_on_exit=True
            )
        ]
    )

    mel_df = pd.DataFrame(
        mel_scaler.fit_transform(mel_values),
        columns=[f"mel_{e}" for e in range(mel_config["n_mels"])],
        index=phoneme_df.index,
    )
    mel_df
    return (mel_df,)


@app.cell
def _(mel_df, run_emd_distance_algorithm):
    mel_emd_differences = run_emd_distance_algorithm(mel_df)
    mel_emd_differences
    return (mel_emd_differences,)


@app.cell
def _(mel_emd_differences):
    mel_emd_differences.mean()
    return


if __name__ == "__main__":
    app.run()
