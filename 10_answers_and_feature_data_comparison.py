import marimo

__generated_with = "0.23.9"
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
    import pandas as pd
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler

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
    from src.plotting import plot_scores
    from src import load_singer_identity_model

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        List,
        MODEL_FOLDER,
        PLOT_FOLDER,
        StandardScaler,
        load_singer_identity_model,
        mo,
        np,
        os,
        pd,
        plot_scores,
    )


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
def _(CSV_FOLDER, os, pd):
    track_df = pd.read_csv(
        os.path.join(
            CSV_FOLDER,
            "LargeDataset",
            "dataset_survey_2_final.csv",
        ),
        index_col="track_id",
    )
    track_df
    return (track_df,)


@app.cell
def _(DATASET_FOLDER, os, pd):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey", "survey_2")


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
def _(surveyQuestions):
    surveyQuestions
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Link Survey Answer to Tracks
    """)
    return


@app.cell
def _(pd, surveyAnswers_, surveyQuestions):
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
    surveyAnswers = surveyAnswers_[~surveyAnswers_.skipped].drop(columns="skipped")
    surveyAnswers
    return (surveyAnswers,)


@app.cell
def _(surveyAnswers):
    RANDOM_CHANCE = len(surveyAnswers[surveyAnswers.answer_1 == "B"]) / len(
        surveyAnswers
    )
    RANDOM_CHANCE
    return (RANDOM_CHANCE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Human Baseline
    """)
    return


@app.cell
def _(surveyAnswers):
    def get_perc_values(questionID: str):
        answers = surveyAnswers[surveyAnswers.questionID == questionID]
        n = len(answers)
        a = None if n == 0 else (len(answers[answers.answer_1 == "A"]) / n)
        b = None if n == 0 else (len(answers[answers.answer_1 == "B"]) / n)
        instruments_on = (
            None if n == 0 else (len(answers[answers.backgroundMusic]) / n)
        )
        agreement = None if n == 0 else max(a, b)
        return {
            "num_answers": n,
            "A_perc": a,
            "B_perc": b,
            "agreement": agreement,
            "instruments_on": instruments_on,
        }

    return (get_perc_values,)


@app.cell
def _(get_perc_values, pd, surveyQuestions):
    surveyQuestions[
        ["num_answers", "A_perc", "B_perc", "agreement", "instruments_on"]
    ] = surveyQuestions.index.to_series().apply(get_perc_values).apply(pd.Series)
    surveyQuestions
    return


@app.cell
def _(surveyQuestions):
    multi_answer_mask = surveyQuestions.num_answers > 1
    HUMAN_BASELINE = surveyQuestions[multi_answer_mask].agreement.mean()
    HUMAN_BASELINE
    return (HUMAN_BASELINE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reusable Agreement Calculation Functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single Feature Scores (Per Feature)
    """)
    return


@app.cell
def _(List, StandardScaler, mo, pd, surveyAnswers):
    def get_scores(x, a_1, a_2):
        scores = {}
        if pd.api.types.is_numeric_dtype(type(x)):
            dist_a_1 = abs(x - a_1)
            dist_a_2 = abs(x - a_2)
            scores["distance"] = (dist_a_2 - dist_a_1) / (dist_a_1 + dist_a_2)
            if dist_a_1 < dist_a_2:
                scores["agreement"] = 1.0  # => agreement
                scores["accuracy"] = 1.0
            elif dist_a_2 < dist_a_1:
                scores["agreement"] = -1.0  # => disagreement
                scores["accuracy"] = 0.0
            else:
                scores["agreement"] = 0.0  # => uncertainty
                scores["accuracy"] = 0.0
        else:
            if a_1 == a_2:
                scores["agreement"] = 0.0  # => uncertainty
                scores["accuracy"] = 0.0
            elif a_1 == x:
                scores["agreement"] = 1.0  # => agreement
                scores["accuracy"] = 1.0
            elif a_2 == x:
                scores["agreement"] = -1.0  # => disagreement
                scores["accuracy"] = 0.0
            else:
                scores["agreement"] = 0.0  # => uncertainty
                scores["accuracy"] = 0.0
        return scores


    def get_local_feature_scores(
        answer,
        feature_df: pd.DataFrame,
        feature_key: str,
    ):
        x = feature_df.loc[answer["track_id_X"]][feature_key]
        a_1 = feature_df.loc[answer["track_id_1"]][feature_key]
        a_2 = feature_df.loc[answer["track_id_2"]][feature_key]
        return get_scores(x, a_1, a_2)


    def get_all_scores(
        feature_df: pd.DataFrame,
        feature_list: List[str],
    ) -> pd.DataFrame:
        columns = {}
        for feat in mo.status.progress_bar(
            feature_list,
            title="Calculating Feature Scores...",
            remove_on_exit=True,
        ):
            feat_scores = surveyAnswers.apply(
                lambda x, f=feat: get_local_feature_scores(x, feature_df, f),
                axis=1,
            )
            feat_df = pd.DataFrame(feat_scores.tolist(), index=surveyAnswers.index)
            for metric in feat_df.columns:
                columns[(metric, feat)] = feat_df[metric]
        agree_df = pd.DataFrame(columns)
        agree_df.columns = pd.MultiIndex.from_tuples(agree_df.columns)
        return agree_df


    def get_mean_values(
        agreement_df: pd.DataFrame, feature_list=None, top_x: int = None
    ) -> dict:
        f_iterator = (
            feature_list if feature_list is not None else agreement_df.columns
        )
        mean_values = {feat: agreement_df[feat].mean() for feat in f_iterator}
        if top_x is not None:
            mean_values = dict(
                sorted(
                    mean_values.items(), key=lambda item: item[1], reverse=True
                )[:top_x]
            )
        return mean_values


    def scale_df(feature_df, columns=None) -> pd.DataFrame:
        scaler = StandardScaler()
        df = feature_df.copy()
        cols = columns if columns is not None else feature_df.columns
        df[cols] = scaler.fit_transform(feature_df[cols])
        return df

    return get_all_scores, get_mean_values, scale_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Set Agreement Scores
    """)
    return


@app.cell
def _():
    from scipy.spatial.distance import (
        euclidean,
        chebyshev,
        cosine,
        minkowski,
        canberra,
    )

    return canberra, chebyshev, cosine, euclidean, minkowski


@app.cell
def _(
    canberra,
    chebyshev,
    cosine,
    euclidean,
    minkowski,
    mo,
    pd,
    surveyAnswers,
):
    minkowski_p = 4


    def get_distance_scores_row(
        answer, feature_df, distance_algorithm
    ) -> pd.Series:
        x = feature_df.loc[answer["track_id_X"]].values
        a_1 = feature_df.loc[answer["track_id_1"]].values
        a_2 = feature_df.loc[answer["track_id_2"]].values
        if distance_algorithm == "euclidean":
            dist_a_1 = euclidean(x, a_1)
            dist_a_2 = euclidean(x, a_2)
        elif distance_algorithm == "chebyshev":
            dist_a_1 = chebyshev(x, a_1)
            dist_a_2 = chebyshev(x, a_2)
        elif distance_algorithm == "cosine":
            dist_a_1 = cosine(x, a_1)
            dist_a_2 = cosine(x, a_2)
        elif distance_algorithm == "minkowski":
            dist_a_1 = minkowski(x, a_1, p=minkowski_p)
            dist_a_2 = minkowski(x, a_2, p=minkowski_p)
        elif distance_algorithm == "canberra":
            dist_a_1 = canberra(x, a_1)
            dist_a_2 = canberra(x, a_2)
        else:
            raise NotImplementedError(
                f"Distance Measure {distance_algorithm} is not implemented yet!"
            )
        dist = (dist_a_2 - dist_a_1) / (dist_a_1 + dist_a_2)
        acc_score = 1.0 if dist_a_1 < dist_a_2 else 0.0
        agg_score = -1.0 if dist_a_2 < dist_a_1 else acc_score
        return pd.Series(
            {"distance": dist, "accuracy": acc_score, "agreement": agg_score}
        )


    distance_algorithms = [
        "euclidean",
        "chebyshev",
        "cosine",
        "minkowski",
        "canberra",
    ]


    def get_global_scores(
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Distance algorithms:
        - euclidean: standard euclidean distance, spreads differences across all features
        - chebyshev: max difference across all dimensions, highly sensitive to single outlier features
        - cosine: measures angle between vectors, captures timbral character regardless of magnitude
        - minkowski: generalization of euclidean with stronger outlier amplification
        - canberra: normalizes per-dimension, amplifies subtle differences in low-valued features
        """
        gda_df = pd.DataFrame()
        for d in mo.status.progress_bar(
            distance_algorithms, title="Calculating GDAs", remove_on_exit=True
        ):
            gda_df[[("distance", d), ("accuracy", d), ("agreement", d)]] = (
                surveyAnswers.apply(
                    lambda x: get_distance_scores_row(x, feature_df, d),
                    axis=1,
                )
            )
        gda_df.columns = pd.MultiIndex.from_tuples(gda_df.columns)
        return gda_df

    return (get_global_scores,)


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
def _(get_all_scores, hl_features, scale_df, track_df):
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
    hl_agreements = get_all_scores(scaled_track_df, hl_features)
    hl_agreements
    return (hl_agreements,)


@app.cell
def _(
    HUMAN_BASELINE,
    PLOT_SAVE_DIR,
    RANDOM_CHANCE,
    get_mean_values,
    hl_agreements,
    os,
    plot_scores,
):
    plot_scores(
        x=get_mean_values(hl_agreements["accuracy"]).values(),
        y=hl_agreements["accuracy"].columns,
        title="High Level Feature Accuracy",
        xlabel="Mean Accuracy (%)",
        random_chance=RANDOM_CHANCE,
        human_baseline=HUMAN_BASELINE,
        save_path=os.path.join(PLOT_SAVE_DIR, "hl_feature_accuracy.png"),
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, get_mean_values, hl_agreements, os, plot_scores):
    plot_scores(
        x=get_mean_values(hl_agreements["agreement"]).values(),
        y=hl_agreements["agreement"].columns,
        title="High Level Feature Agreement",
        xlabel="Mean Agreement Score",
        save_path=os.path.join(PLOT_SAVE_DIR, "hl_feature_agreement.png"),
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, get_mean_values, hl_agreements, os, plot_scores):
    plot_scores(
        x=get_mean_values(hl_agreements["distance"]).values(),
        y=hl_agreements["distance"].columns,
        title="High Level Feature Distance Differences",
        xlabel="Mean Distance Difference: dist(x,b)-dist(x,a) / dist(x,a)+dist(x,b)",
        save_path=os.path.join(PLOT_SAVE_DIR, "hl_feature_distance.png"),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Default Embedding Agreement
    Calculate the embeddings using the previously used embedding model:
    - MelSpectrogramEncoder from [pip](https://speechbrain.readthedocs.io/en/latest/installation.html), thanks to speechbrain
    - Encoder Model from [HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb-mel-spec), thanks to speechbrain

    Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., Subakan, C., Dawalatabad, N., Heba, A., Zhong, J., Chou, J.-C., Yeh, S.-L., Fu, S.-W., Liao, C.-F., Rastorgueva, E., Grondin, F., Aris, W., Na, H., Gao, Y., … Bengio, Y. (2021). SpeechBrain: A General-Purpose Speech Toolkit.
    """)
    return


@app.cell
def _():
    import torch
    import torchaudio
    from speechbrain.inference.encoders import MelSpectrogramEncoder

    from src.utils import get_trimmed_audio

    return MelSpectrogramEncoder, get_trimmed_audio, torch


@app.cell
def _(MelSpectrogramEncoder):
    encoder = MelSpectrogramEncoder.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb-mel-spec"
    )
    SAMPLE_RATE = 16_000
    return SAMPLE_RATE, encoder


@app.cell
def _(SAMPLE_RATE, get_trimmed_audio):
    def get_embedding(song_path, enc):
        trimmed_audio = get_trimmed_audio(song_path, sr=SAMPLE_RATE)
        return enc.encode_waveform(trimmed_audio).cpu().numpy().squeeze()

    return (get_embedding,)


@app.cell
def _(encoder, get_embedding, np, pd, scale_df, track_df):
    embedding_df = pd.DataFrame(
        np.stack(
            track_df.song_path.apply(lambda x: get_embedding(x, encoder)).values
        ),
        columns=[f"emb_{e}" for e in range(192)],
        index=track_df.index,
    )
    embedding_df = scale_df(embedding_df)
    embedding_df
    return (embedding_df,)


@app.cell
def _(embedding_df, get_global_scores):
    embedding_gda_df = get_global_scores(embedding_df)
    embedding_gda_df["accuracy"].describe()
    return (embedding_gda_df,)


@app.cell
def _(
    HUMAN_BASELINE,
    PLOT_SAVE_DIR,
    RANDOM_CHANCE,
    embedding_gda_df,
    os,
    plot_scores,
):
    plot_scores(
        x=embedding_gda_df["accuracy"].mean(),
        y=embedding_gda_df["accuracy"].columns,
        title="Embedding Accuracy",
        random_chance=RANDOM_CHANCE,
        human_baseline=HUMAN_BASELINE,
        xlabel="Mean Accuracy (%)",
        ylabel="Distance Algorithm",
        save_path=os.path.join(PLOT_SAVE_DIR, "emb_gda_accuracy.png"),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Fine Tuned Embedding Agreement
    """)
    return


@app.cell
def _():
    """
    finetuned_encoder_path = os.path.join(
        MODEL_FOLDER,
        "finetuned_encoder",
        f"spkrec-ecapa-voxceleb-mel-spec_shuffle_32_0.3_40_1e-05.pt",
    )
    encoder.hparams.embedding_model.load_state_dict(
        torch.load(finetuned_encoder_path)
    )

    ft_embedding_df = pd.DataFrame(
        np.stack(
            track_df.song_path.apply(lambda x: get_embedding(x, encoder)).values
        ),
        columns=[f"emb_{e}" for e in range(192)],
        index=track_df.index,
    )
    ft_embedding_df = scale_df(ft_embedding_df)
    ft_embedding_df

    ft_embedding_gda_df = get_global_scores(ft_embedding_df)
    ft_embedding_gda_df["accuracy"].describe()


    plot_scores(
        x=ft_embedding_gda_df["accuracy"].mean(),
        y=ft_embedding_gda_df["accuracy"].columns,
        title="Fine Tuned Embedding Accuracy",
        random_chance=RANDOM_CHANCE,
        human_baseline=HUMAN_BASELINE,
        xlabel="Mean Accuracy (%)",
        ylabel="Distance Algorithm",
        save_path=os.path.join(PLOT_SAVE_DIR, "ft_emb_gda_accuracy.png"),
    )
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Singer Identity Model Agreement
    """)
    return


@app.cell
def _(MODEL_FOLDER, load_singer_identity_model, os):
    singer_ID_model_path = os.path.join(MODEL_FOLDER, "singer-identity", "byol")
    sID_SAMPLE_RATE=44100
    singer_ID_model = load_singer_identity_model(singer_ID_model_path, input_sr=sID_SAMPLE_RATE)
    singer_ID_model.eval()
    return sID_SAMPLE_RATE, singer_ID_model


@app.cell
def _(get_trimmed_audio, sID_SAMPLE_RATE):
    def get_singer_ID_embedding(song_path, enc):
        trimmed_audio = get_trimmed_audio(song_path, sr=sID_SAMPLE_RATE)
        return enc(trimmed_audio).cpu().detach().squeeze()

    return (get_singer_ID_embedding,)


@app.cell
def _(
    get_singer_ID_embedding,
    np,
    pd,
    scale_df,
    singer_ID_model,
    torch,
    track_df,
):
    with torch.no_grad():
        sID_embedding_df = pd.DataFrame(
            np.stack(
                track_df.song_path.apply(lambda x: get_singer_ID_embedding(x, singer_ID_model)).values
            ),
            columns=[f"emb_{e}" for e in range(1000)],
            index=track_df.index,
        )
    sID_embedding_df = scale_df(sID_embedding_df)
    sID_embedding_df
    return (sID_embedding_df,)


@app.cell
def _(get_global_scores, sID_embedding_df):
    sID_embedding_gda_df = get_global_scores(sID_embedding_df)
    sID_embedding_gda_df["accuracy"].describe()
    return (sID_embedding_gda_df,)


@app.cell
def _(
    HUMAN_BASELINE,
    PLOT_SAVE_DIR,
    RANDOM_CHANCE,
    os,
    plot_scores,
    sID_embedding_gda_df,
):
    plot_scores(
        x=sID_embedding_gda_df["accuracy"].mean(),
        y=sID_embedding_gda_df["accuracy"].columns,
        title="Singer ID Embedding Accuracy",
        random_chance=RANDOM_CHANCE,
        human_baseline=HUMAN_BASELINE,
        xlabel="Mean Accuracy (%)",
        ylabel="Distance Algorithm",
        save_path=os.path.join(PLOT_SAVE_DIR, "sID_emb_gda_accuracy.png"),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # GeMAPS Feature Set
    """)
    return


@app.cell
def _():
    import opensmile

    smile_gemaps = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return opensmile, smile_gemaps


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
def _(gemaps_features_df, get_all_scores):
    gemaps_agreements = get_all_scores(
        gemaps_features_df, gemaps_features_df.columns
    )
    gemaps_agreements
    return (gemaps_agreements,)


@app.cell
def _(
    HUMAN_BASELINE,
    PLOT_SAVE_DIR,
    RANDOM_CHANCE,
    gemaps_agreements,
    get_mean_values,
    os,
    plot_scores,
):
    TOP_X = 15
    top_gemaps_score_values = get_mean_values(
        gemaps_agreements["accuracy"], top_x=TOP_X
    )

    plot_scores(
        x=top_gemaps_score_values.values(),
        y=top_gemaps_score_values.keys(),
        title=f"GeMAPS Single Feature Accuracy (Top {TOP_X})",
        random_chance=RANDOM_CHANCE,
        human_baseline=HUMAN_BASELINE,
        xlabel="Mean Accuracy (%)",
        save_path=os.path.join(PLOT_SAVE_DIR, "gemaps_single_accuracy.png"),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature Set Agreement
    """)
    return


@app.cell
def _(gemaps_features_df, get_global_scores):
    gemaps_gda_df = get_global_scores(gemaps_features_df)
    gemaps_gda_df["accuracy"].describe()
    return (gemaps_gda_df,)


@app.cell
def _(
    HUMAN_BASELINE,
    PLOT_SAVE_DIR,
    RANDOM_CHANCE,
    gemaps_gda_df,
    os,
    plot_scores,
):
    plot_scores(
        x=gemaps_gda_df["accuracy"].mean(),
        y=gemaps_gda_df["accuracy"].columns,
        title="All GeMAPS Features Accuracy",
        random_chance=RANDOM_CHANCE,
        human_baseline=HUMAN_BASELINE,
        xlabel="Mean Accuracy (%)",
        ylabel="Distance Algorithm",
        save_path=os.path.join(PLOT_SAVE_DIR, "gemaps_all_accuracy.png"),
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
def _(compare_features_df, get_global_scores):
    compare_gda_df = get_global_scores(compare_features_df)
    compare_gda_df["accuracy"].describe()
    return (compare_gda_df,)


@app.cell
def _(
    HUMAN_BASELINE,
    PLOT_SAVE_DIR,
    RANDOM_CHANCE,
    compare_gda_df,
    os,
    plot_scores,
):
    plot_scores(
        x=compare_gda_df["accuracy"].mean(),
        y=compare_gda_df["accuracy"].columns,
        title="All ComParE Features Accuracy",
        random_chance=RANDOM_CHANCE,
        human_baseline=HUMAN_BASELINE,
        xlabel="Mean Accuracy (%)",
        ylabel="Distance Algorithm",
        save_path=os.path.join(PLOT_SAVE_DIR, "compare_all_accuracy.png"),
    )
    return


if __name__ == "__main__":
    app.run()
