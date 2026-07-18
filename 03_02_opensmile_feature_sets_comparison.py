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
        get_global_distance_scores,
        scale_df,
        get_distance_row,
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
        get_distance_row,
        get_global_distance_scores,
        get_trimmed_audio,
        load_survey_data,
        mo,
        np,
        opensmile,
        os,
        pd,
        plot_correlation_bar,
        plot_correlation_scatter,
        plt,
        scale_df,
        sns,
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
    return (hl_distances,)


@app.cell
def _(mo):
    mo.md(r"""
    # GeMAPS Feature Set
    """)
    return


@app.cell
def _(opensmile):
    smile_gemaps = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return (smile_gemaps,)


@app.cell
def _(SAMPLE_RATE, get_trimmed_audio):
    def get_feature_set(song_path, sm):
        trimmed_audio = get_trimmed_audio(song_path, sr=SAMPLE_RATE)
        return sm.process_signal(trimmed_audio, SAMPLE_RATE).values[0]

    return


@app.cell
def _(DATASET_FOLDER, os):
    gemaps_feature_path = os.path.join(
        DATASET_FOLDER, "fma_large_feature_sets", "survey_2_gemaps.npy"
    )
    return (gemaps_feature_path,)


@app.cell
def _():
    """
    gemaps_features = pd.DataFrame(
        track_df.song_path.apply(
            lambda x: get_feature_set(x, smile_gemaps)
        ).tolist(),
        columns=smile_gemaps.feature_names,
        index=track_df.index,
    )
    gemaps_features


    with open(gemaps_feature_path, "wb") as npyfile:
        np.save(npyfile, gemaps_features.values)
    """
    return


@app.cell
def _(gemaps_feature_path, np, pd, scale_df, smile_gemaps, track_df):
    gemaps_features_df = pd.DataFrame(
        np.load(gemaps_feature_path),
        columns=smile_gemaps.feature_names,
        index=track_df.index,
    )
    gemaps_features_df = scale_df(gemaps_features_df)
    gemaps_features_df
    return (gemaps_features_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single Feature Agreement
    """)
    return


@app.cell
def _(gemaps_features_df, get_all_distance_differences, questions_df):
    gemaps_distances = get_all_distance_differences(
        gemaps_features_df, gemaps_features_df.columns, questions_df
    )
    gemaps_distances
    return (gemaps_distances,)


@app.cell
def _(gemaps_distances, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="GeMAPS Feature Correlations (Randomized)",
        feature_df=gemaps_distances[questions_df.randomized],
        target_feature=questions_df[questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(gemaps_distances, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="GeMAPS Feature Correlations  (Max Entropy)",
        feature_df=gemaps_distances[~questions_df.randomized],
        target_feature=questions_df[~questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(gemaps_distances, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="GeMAPS Feature Correlations (All)",
        feature_df=gemaps_distances,
        target_feature=questions_df["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(gemaps_features_df, plot_correlation_bar, track_df):
    plot_correlation_bar(
        title="GeMAPS Feature Correlations (Speaker Gender)",
        feature_df=gemaps_features_df,
        target_feature=track_df["pred_p_male"],
        top_x=10,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gender dependent
    """)
    return


@app.cell
def _(np, plt, questions_df):
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    bin_labels = [
        "Only Female Voices",
        "Female Reference Voices, Male Target Voice",
        "Mixed Reference Voices",
        "Male Reference Voices, Female Target Voice",
        "Only Male Voices",
    ]
    counts, bin_edges = np.histogram(
        questions_df["gender_distribution"], bins=bins
    )
    plt.figure(figsize=(10, 4), dpi=150)
    bars = plt.barh(bin_labels, counts, color="skyblue", edgecolor="black")
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=10,
        )
    plt.xlabel("Number of Questions")
    # plt.ylabel("Gender Distribution Categories")
    plt.title("Gender Distribution Counts in Survey")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(questions_df):
    gender_m_mask = questions_df["gender_distribution"] >= 0.75
    gender_f_mask = questions_df["gender_distribution"] <= 0.25
    gender_mixed_mask = questions_df["gender_distribution"] == 0.5
    return gender_f_mask, gender_m_mask, gender_mixed_mask


@app.cell
def _(gemaps_distances, gender_m_mask, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="GeMAPS Feature Correlations (Male only)",
        feature_df=gemaps_distances[gender_m_mask],
        target_feature=questions_df[gender_m_mask]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_distances,
    gender_m_mask,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        title="GeMAPS F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope (Male Only)",
        feature_name="F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
        y=gemaps_distances[gender_m_mask][
            "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope"
        ],
        x=questions_df[gender_m_mask]["A_perc"],
        plot_dir=PLOT_SAVE_DIR,
    )
    return


@app.cell
def _(gemaps_distances, gender_f_mask, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="GeMAPS Feature Correlations (Female only)",
        feature_df=gemaps_distances[gender_f_mask],
        target_feature=questions_df[gender_f_mask]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_distances,
    gender_f_mask,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        title="GeMAPS StddevVoicedSegmentLengthSec (Female Only)",
        feature_name="StddevVoicedSegmentLengthSec",
        y=gemaps_distances[gender_f_mask]["StddevVoicedSegmentLengthSec"],
        x=questions_df[gender_f_mask]["A_perc"],
        plot_dir=PLOT_SAVE_DIR,
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, gemaps_distances, plot_correlation_scatter, questions_df):
    plot_correlation_scatter(
        title="GeMAPS F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
        feature_name="F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
        y=gemaps_distances["F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope"],
        x=questions_df["A_perc"],
        plot_dir=PLOT_SAVE_DIR,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature Set Agreement
    """)
    return


@app.cell
def _(gemaps_features_df, get_global_distance_scores, questions_df):
    gemaps_gda_df = get_global_distance_scores(
        gemaps_features_df, questions_df
    )
    gemaps_gda_df
    return (gemaps_gda_df,)


@app.cell
def _(gemaps_gda_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="GeMAPS Feature Set Correlations (All)",
        feature_df=gemaps_gda_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, gemaps_gda_df, plot_correlation_scatter, questions_df):
    plot_correlation_scatter(
        title="GeMAPS Feature Set (Canberra)",
        feature_name="Feature_Set_Canberra",
        y=gemaps_gda_df["distance_canberra"],
        x=questions_df["A_perc"],
        plot_dir=PLOT_SAVE_DIR,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backwards Stepwise Regression
    """)
    return


@app.cell
def _(questions_df):
    questions_df
    return


@app.cell
def _():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score, KFold
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    return (sm,)


@app.cell
def _(gemaps_features_df, np, questions_df):
    def get_feature_differences(question):
        X_features = gemaps_features_df.loc[question["X"]].values
        A_features = gemaps_features_df.loc[question["A"]].values
        B_features = gemaps_features_df.loc[question["B"]].values
        XA_diff = X_features - A_features
        XB_diff = X_features - B_features

        XAB_diff = abs(XB_diff) - abs(XA_diff)
        return XAB_diff

    gemaps_feature_differences = np.stack(
        questions_df.apply(get_feature_differences, axis=1)
    )
    gemaps_feature_differences.shape
    return (gemaps_feature_differences,)


@app.cell
def _(gemaps_feature_differences, np, pd, plt, smile_gemaps, sns):
    gemaps_feature_corr = pd.DataFrame(
        gemaps_feature_differences, columns=smile_gemaps.feature_names
    ).corr()
    mask = np.triu(np.ones_like(gemaps_feature_corr, dtype=bool))

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        gemaps_feature_corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    return (gemaps_feature_corr,)


@app.cell
def _(gemaps_feature_corr, np, smile_gemaps):
    reduced_selection = [0, 1, 10, 30, 31, 32, 33, 34, 35, 78, 79, 80]

    corr_threshold = 0.7
    for i in range(len(smile_gemaps.feature_names)):
        if i in reduced_selection:
            continue

        for j in reduced_selection:
            if gemaps_feature_corr.values[i, j] >= corr_threshold:
                break
        else:
            reduced_selection.append(i)

    reduced_feature_names = np.array(smile_gemaps.feature_names)[
        reduced_selection
    ]
    len(reduced_feature_names)
    return reduced_feature_names, reduced_selection


@app.cell
def _(
    gemaps_feature_differences,
    questions_df,
    reduced_feature_names,
    reduced_selection,
    sm,
):
    def backward_stepwise_regression(X, y, feature_names, alpha=0.05):
        """
        Perform backward stepwise regression
        """
        # Start with all features
        current_features = list(range(X.shape[1]))
        current_pvalues = None
        removed_features = []

        while True:
            # Fit model with current features
            X_current = X[:, current_features]
            X_sm = sm.add_constant(X_current)
            model = sm.OLS(y, X_sm).fit()

            # Get p-values (excluding intercept)
            pvalues = model.pvalues[1:]

            # Find feature with highest p-value
            max_p = pvalues.max()
            max_p_idx = pvalues.argmax()

            if max_p > alpha and len(current_features) > 1:
                # Remove feature
                removed_features.append((current_features[max_p_idx], max_p))
                del current_features[max_p_idx]
            else:
                break

        # Final model
        X_final = X[:, current_features]
        X_sm = sm.add_constant(X_final)
        final_model = sm.OLS(y, X_sm).fit()

        return final_model, current_features, removed_features

    final_model, selected_features, removed = backward_stepwise_regression(
        gemaps_feature_differences[:, reduced_selection],
        questions_df["A_perc"].values,
        reduced_feature_names,
    )
    print(final_model.summary())
    print(
        f"Selected features: {[str(reduced_feature_names[i]) for i in selected_features]}"
    )
    return


@app.cell
def _():
    """
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.299
    Model:                            OLS   Adj. R-squared:                  0.266
    Method:                 Least Squares   F-statistic:                     8.863
    Date:                Thu, 16 Jul 2026   Prob (F-statistic):           4.51e-10
    Time:                        16:45:44   Log-Likelihood:                 21.640
    No. Observations:                 175   AIC:                            -25.28
    Df Residuals:                     166   BIC:                             3.203
    Df Model:                           8
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.4664      0.017     27.280      0.000       0.433       0.500
    x1             0.0706      0.016      4.513      0.000       0.040       0.101
    x2             0.0459      0.016      2.952      0.004       0.015       0.077
    x3            -0.0397      0.016     -2.442      0.016      -0.072      -0.008
    x4             0.0516      0.014      3.654      0.000       0.024       0.080
    x5             0.0511      0.017      3.057      0.003       0.018       0.084
    x6             0.0324      0.015      2.193      0.030       0.003       0.062
    x7            -0.0501      0.018     -2.831      0.005      -0.085      -0.015
    x8             0.0424      0.017      2.516      0.013       0.009       0.076
    ==============================================================================
    Omnibus:                        5.796   Durbin-Watson:                   1.977
    Prob(Omnibus):                  0.055   Jarque-Bera (JB):                3.273
    Skew:                          -0.099   Prob(JB):                        0.195
    Kurtosis:                       2.360   Cond. No.                         1.84
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Selected features: ['F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'mfcc1_sma3_stddevNorm', 'mfcc3_sma3_amean', 'mfcc4_sma3_amean', 'F1frequency_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean', 'F2bandwidth_sma3nz_amean', 'loudnessPeaksPerSec']

    """
    return


@app.cell
def _(PLOT_SAVE_DIR, gemaps_distances, plot_correlation_scatter, questions_df):
    plot_correlation_scatter(
        title="GeMAPS F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
        feature_name="F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
        y=gemaps_distances["F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope"],
        x=questions_df["A_perc"],
        plot_dir=PLOT_SAVE_DIR,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # ComParE Feature Set
    """)
    return


@app.cell
def _(opensmile):
    smile_compare = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return (smile_compare,)


@app.cell
def _(DATASET_FOLDER, os):
    compare_feature_path = os.path.join(
        DATASET_FOLDER, "fma_large_feature_sets", "survey_2_compare.npy"
    )
    return (compare_feature_path,)


@app.cell
def _():
    """
    compare_features = pd.DataFrame(
        track_df.song_path.apply(
            lambda x: get_feature_set(x, smile_compare)
        ).tolist(),
        columns=smile_compare.feature_names,
        index=track_df.index,
    )
    compare_features


    with open(compare_feature_path, "wb") as npyfile:
        np.save(npyfile, compare_features.values)
    """
    return


@app.cell
def _(compare_feature_path, np, pd, scale_df, smile_compare, track_df):
    compare_features_df = pd.DataFrame(
        np.load(compare_feature_path),
        columns=smile_compare.feature_names,
        index=track_df.index,
    )
    compare_features_df = scale_df(compare_features_df)
    compare_features_df
    return (compare_features_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Single Feature Agreement
    """)
    return


@app.cell
def _():
    """
    compare_agreements = get_all_scores(
        compare_features_df, compare_features_df.columns
    )
    top_compare_score_values = get_mean_values(compare_agreements, top_x=15)

    plot_scores(
        x=top_compare_score_values.values(),
        y=top_compare_score_values.keys(),
        title=f"ComParE Single Feature Accuracy (Top {TOP_X})",
        random_chance=RANDOM_CHANCE,
        xlabel="Mean Accuracy (%)",
        save_path=os.path.join(PLOT_SAVE_DIR, "compare_single_accuracy.png"),
    )
    """
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature Set Agreement
    """)
    return


@app.cell
def _(compare_features_df, get_global_distance_scores, questions_df):
    compare_gda_df = get_global_distance_scores(
        compare_features_df, questions_df
    )
    compare_gda_df
    return (compare_gda_df,)


@app.cell
def _(compare_gda_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="ComParE Feature Set Correlations",
        feature_df=compare_gda_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, compare_gda_df, plot_correlation_scatter, questions_df):
    plot_correlation_scatter(
        title="ComParE Feature Set (Cosine)",
        feature_name="ComParE_Feature_Set_Cosine",
        x=questions_df["A_perc"],
        y=compare_gda_df["distance_cosine"],
        plot_dir=PLOT_SAVE_DIR,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Voice Quality High Level Features

    Use only cosine distances
    """)
    return


@app.cell
def _(compare_features_df, gemaps_features_df):
    from src.statistics.opensmile_mapping import (
        FEATURE_MAP,
        convert_to_voice_quality_features,
    )

    VOICE_QUALITY_FEATURES = list(FEATURE_MAP.keys())
    voice_quality_df = convert_to_voice_quality_features(
        gemaps_features_df, compare_features_df
    )
    voice_quality_df
    return VOICE_QUALITY_FEATURES, voice_quality_df


@app.cell
def _(
    VOICE_QUALITY_FEATURES,
    get_distance_row,
    mo,
    pd,
    questions_df,
    voice_quality_df,
):
    vq_distance_diff_df = pd.DataFrame()
    for vq_feature in mo.status.progress_bar(
        VOICE_QUALITY_FEATURES,
        title="Calculating Global Distances",
        remove_on_exit=True,
    ):
        vq_distance_diff_df[vq_feature] = questions_df.apply(
            lambda x: get_distance_row(
                x, voice_quality_df[vq_feature], "cosine"
            ),
            axis=1,
        )
    vq_distance_diff_df
    return (vq_distance_diff_df,)


@app.cell
def _(plot_correlation_bar, questions_df, vq_distance_diff_df):
    plot_correlation_bar(
        title="Correlation of Voice Quality Features with Subjective Similarity Ratings",
        feature_df=vq_distance_diff_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(
    gender_mixed_mask,
    hl_distances,
    plot_correlation_bar,
    vq_distance_diff_df,
):
    plot_correlation_bar(
        title="Voice Quality Features Comparison with Gender (Mixed Gender)",
        feature_df=vq_distance_diff_df[gender_mixed_mask],
        target_feature=hl_distances[gender_mixed_mask]["pred_p_male"],
        top_x=10,
    )
    return


@app.cell
def _():
    """
    plot_correlation_bar(
        title="Voice Quality Features Comparison (Mixed Gender)",
        feature_df=vq_distance_diff_df[gender_mixed_mask],
        target_feature=questions_df[gender_mixed_mask]["A_perc"],
        top_x=10,
    )
    """
    return


@app.cell
def _(gender_m_mask, plot_correlation_bar, questions_df, vq_distance_diff_df):
    plot_correlation_bar(
        title="Voice Quality Features Comparison (Male only)",
        feature_df=vq_distance_diff_df[gender_m_mask],
        target_feature=questions_df[gender_m_mask]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(gender_f_mask, plot_correlation_bar, questions_df, vq_distance_diff_df):
    plot_correlation_bar(
        title="Voice Quality Features Comparison (Female only)",
        feature_df=vq_distance_diff_df[gender_f_mask],
        target_feature=questions_df[gender_f_mask]["A_perc"],
        top_x=10,
    )
    return


if __name__ == "__main__":
    app.run()
