import marimo

__generated_with = "0.23.14"
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

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from scipy.stats import wasserstein_distance_nd

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
    from src.phoneme_extractor.phoneme_extractor import (
        load_data as load_phoneme_data,
    )
    from src.statistics.plotting import plot_scores
    from src.survey_dataset_helpers import load_survey_data
    from src.statistics.feature_correlation import (
        scale_df,
        get_all_distance_differences,
    )
    from src.statistics.plotting import (
        plot_correlation_bar,
        plot_correlation_scatter,
    )

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        PLOT_FOLDER,
        get_all_distance_differences,
        librosa,
        load_phoneme_data,
        load_survey_data,
        mo,
        np,
        os,
        pd,
        plot_correlation_bar,
        plot_correlation_scatter,
        scale_df,
        torch,
        wasserstein_distance_nd,
    )


@app.cell
def _(PLOT_FOLDER, os):
    PLOT_SAVE_DIR = os.path.join(PLOT_FOLDER, "survey_2")

    return (PLOT_SAVE_DIR,)


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
def _(CSV_FOLDER, DATASET_FOLDER, load_survey_data, os):
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
    SURVEY_DATA = load_survey_data(CSV_PATHS)
    questions_df = SURVEY_DATA["questions_df"]
    answers_df = SURVEY_DATA["answers_df"]
    participants_df = SURVEY_DATA["participants_df"]
    songs_df = SURVEY_DATA["songs_df"]
    human_agreement = SURVEY_DATA["human_agreement"]
    answer_a_b_ratio = SURVEY_DATA["answer_a_b_ratio"]
    track_df = SURVEY_DATA["track_df"]
    return questions_df, songs_df, track_df


@app.cell
def _(
    MIN_PHONEME_CONFIDENCE,
    MIN_PHONEME_DURATION_MS,
    load_phoneme_data,
    phoneme_save_path,
    songs_df,
):
    phoneme_df, phonemes = load_phoneme_data(phoneme_save_path)
    phoneme_df = phoneme_df.rename(columns={"file_id": "trackID"})

    phoneme_mask = (phoneme_df.duration_ms >= MIN_PHONEME_DURATION_MS) & (
        phoneme_df.confidence >= MIN_PHONEME_CONFIDENCE
    )
    phonemes = phonemes[phoneme_mask]
    phoneme_df = phoneme_df[phoneme_mask]

    trackID_mask = phoneme_df.trackID.isin(songs_df.trackID.unique())
    phoneme_df = phoneme_df[trackID_mask]
    phonemes = phonemes[trackID_mask]

    phoneme_df.groupby("trackID").size().agg(["min", "mean"])
    return phoneme_df, phonemes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reusable Functions
    """)
    return


@app.cell
def _(mo, pd, phoneme_df, questions_df, wasserstein_distance_nd):
    def get_emd_distance_diff(x, a_1, a_2) -> float:
        """Get the differences of EMD-Distances for a triplet of distributions
        x, a_1 and a_2 should be a distribution as nd-vectors of feature values
        """
        emd_1 = wasserstein_distance_nd(x, a_1)
        emd_2 = wasserstein_distance_nd(x, a_2)
        norm_dist = (emd_2 - emd_1) / (emd_1 + emd_2)
        return (norm_dist + 1) / 2


    def get_phoneme_features_per_track(trackID, feature_df):
        return feature_df[phoneme_df.trackID == trackID]


    def get_local_feature_emd_distance_diff(
        question,
        feature_df: pd.DataFrame,
    ) -> float:
        x = get_phoneme_features_per_track(question["X"], feature_df)
        a_1 = get_phoneme_features_per_track(question["A"], feature_df)
        a_2 = get_phoneme_features_per_track(question["B"], feature_df)
        return get_emd_distance_diff(x, a_1, a_2)


    def run_emd_distance_diff_algorithm(feature_df):
        columns = {}
        for feat in mo.status.progress_bar(
            feature_df.columns,
            title="Calculating EMD Distance Algorithm...",
            remove_on_exit=True,
        ):
            feat_scores = questions_df.apply(
                lambda x, f=feat: get_local_feature_emd_distance_diff(
                    x, feature_df[f]
                ),
                axis=1,
            )
            columns[feat] = feat_scores.to_list()
        return pd.DataFrame(columns, index=questions_df.index)

    return (run_emd_distance_diff_algorithm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MFCCs
    """)
    return


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
def _(get_mfcc_stats, mfcc_config, mo, np, pd, phoneme_df, phonemes, scale_df):
    mfcc_df = scale_df(
        pd.DataFrame(
            np.stack(
                [
                    get_mfcc_stats(x, **mfcc_config)
                    for x in mo.status.progress_bar(
                        phonemes, title="Calculating MFCCs...", remove_on_exit=True
                    )
                ]
            ),
            columns=[f"mfcc_{e}" for e in range(mfcc_config["n_mfcc"])],
            index=phoneme_df.index,
        )
    )
    mfcc_df
    return (mfcc_df,)


@app.cell
def _(mfcc_df, run_emd_distance_diff_algorithm):
    mfcc_emd_distance_diff_df = run_emd_distance_diff_algorithm(mfcc_df)
    mfcc_emd_distance_diff_df
    return (mfcc_emd_distance_diff_df,)


@app.cell
def _(mfcc_emd_distance_diff_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="MFCC EMD Correlations (Randomized)",
        feature_df=mfcc_emd_distance_diff_df[questions_df.randomized],
        target_feature=questions_df[questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(mfcc_emd_distance_diff_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="MFCC EMD Correlations (Max Entropy)",
        feature_df=mfcc_emd_distance_diff_df[~questions_df.randomized],
        target_feature=questions_df[~questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(mfcc_emd_distance_diff_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="MFCC EMD Correlations (All)",
        feature_df=mfcc_emd_distance_diff_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    mfcc_emd_distance_diff_df,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        title="MFCC 4th Coefficient",
        feature_name="MFCC_4th_Coedd",
        x=questions_df["A_perc"],
        y=mfcc_emd_distance_diff_df["mfcc_4"],
        plot_dir=PLOT_SAVE_DIR,
    )
    return


@app.cell
def _(
    get_all_distance_differences,
    mfcc_emd_distance_diff_df,
    plot_correlation_bar,
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
    hl_distances = get_all_distance_differences(
        track_df, hl_features, questions_df
    )
    hl_distances

    plot_correlation_bar(
        title="MFCC EMD Correlations (Speaker Gender)",
        feature_df=mfcc_emd_distance_diff_df,
        target_feature=hl_distances["pred_p_male"],
        top_x=10,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gender Dependent
    """)
    return


@app.cell
def _(questions_df):
    gender_m_mask = questions_df["gender_distribution"] == 1.0
    gender_f_mask = questions_df["gender_distribution"] == 0.0
    gender_mixed_mask = questions_df["gender_distribution"] == 0.5
    return gender_f_mask, gender_m_mask, gender_mixed_mask


@app.cell
def _(
    gender_m_mask,
    mfcc_emd_distance_diff_df,
    plot_correlation_bar,
    questions_df,
):
    plot_correlation_bar(
        title="MFCC EMD Feature Correlations (Male only)",
        feature_df=mfcc_emd_distance_diff_df[gender_m_mask],
        target_feature=questions_df[gender_m_mask]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(
    gender_f_mask,
    mfcc_emd_distance_diff_df,
    plot_correlation_bar,
    questions_df,
):
    plot_correlation_bar(
        title="MFCC EMD Feature Correlations (Female only)",
        feature_df=mfcc_emd_distance_diff_df[gender_f_mask],
        target_feature=questions_df[gender_f_mask]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(
    gender_mixed_mask,
    mfcc_emd_distance_diff_df,
    plot_correlation_bar,
    questions_df,
):
    plot_correlation_bar(
        title="MFCC EMD Feature Correlations (Mixed Gender)",
        feature_df=mfcc_emd_distance_diff_df[gender_mixed_mask],
        target_feature=questions_df[gender_mixed_mask]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mel Frequencies
    """)
    return


@app.cell
def _(librosa):
    mel_config = {
        "fmin": 50,
        "fmax": 8_000,
        "n_mels": 128,
    }
    mel_centers = librosa.mel_frequencies(
        n_mels=mel_config["n_mels"],
        fmin=mel_config["fmin"],
        fmax=mel_config["fmax"],
    )
    mel_centers[:10]
    return mel_centers, mel_config


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
    get_mel_frequencies,
    mel_config,
    mo,
    np,
    pd,
    phoneme_df,
    phonemes,
    scale_df,
):
    mel_df = scale_df(
        pd.DataFrame(
            np.stack(
                [
                    get_mel_frequencies(x, **mel_config)
                    for x in mo.status.progress_bar(
                        phonemes, title="Calculating Mels...", remove_on_exit=True
                    )
                ]
            ),
            columns=[f"mel_{e}" for e in range(mel_config["n_mels"])],
            index=phoneme_df.index,
        )
    )
    mel_df
    return (mel_df,)


@app.cell
def _(mel_df, run_emd_distance_diff_algorithm):
    mel_emd_distance_diff_df = run_emd_distance_diff_algorithm(mel_df)
    mel_emd_distance_diff_df
    return (mel_emd_distance_diff_df,)


@app.cell
def _(mel_emd_distance_diff_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="Mel EMD Correlations (Randomized)",
        feature_df=mel_emd_distance_diff_df[questions_df.randomized],
        target_feature=questions_df[questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(mel_emd_distance_diff_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="Mel EMD Correlations (Max Entropy)",
        feature_df=mel_emd_distance_diff_df[~questions_df.randomized],
        target_feature=questions_df[~questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(mel_emd_distance_diff_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="Mel EMD Correlations (All)",
        feature_df=mel_emd_distance_diff_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    mel_centers,
    mel_emd_distance_diff_df,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        title=f"Mel 4th Band ({mel_centers[4]:.1f} Hz)",
        feature_name="Mel_4th_Band",
        x=questions_df["A_perc"],
        y=mel_emd_distance_diff_df["mel_4"],
        plot_dir=PLOT_SAVE_DIR,
    )
    return


if __name__ == "__main__":
    app.run()
