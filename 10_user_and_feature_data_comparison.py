import marimo

__generated_with = "0.23.8"
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
    from typing import List, Literal
    import matplotlib.pyplot as plt

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
        List,
        Literal,
        MODEL_FOLDER,
        mo,
        np,
        os,
        pd,
        sns,
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
def _(List, mo, pd, surveyAnswers):
    def getAgreementScore(x, a_1, a_2, use_distance: bool):
        if pd.api.types.is_numeric_dtype(type(x)):
            dist_a_1 = abs(x - a_1)
            dist_a_2 = abs(x - a_2)
            if use_distance:
                return (dist_a_2 - dist_a_1) / (dist_a_1 + dist_a_2)
            else:
                if dist_a_1 < dist_a_2:
                    return 1.0  # => agreement
                if dist_a_2 < dist_a_1:
                    return -1.0  # => disagreement
                else:
                    return 0.0  # => uncertainty
        else:
            if a_1 == a_2:
                return 0.0  # => uncertainty
            elif a_1 == x:
                return 1.0  # => agreement
            elif a_2 == x:
                return -1.0  # => disagreement
            else:
                return 0.0  # => unvertainty (all are different)


    def getLocalFeatureAgreement(
        answer,
        feature_df: pd.DataFrame,
        feature_key: str,
        use_distance: bool = False,
    ):
        x = feature_df.loc[answer["track_id_X"]][feature_key]
        a_1 = feature_df.loc[answer["track_id_1"]][feature_key]
        a_2 = feature_df.loc[answer["track_id_2"]][feature_key]
        return getAgreementScore(x, a_1, a_2, use_distance)


    def getAllAgreements(
        feature_df: pd.DataFrame,
        feature_list: List[str],
        add_distance_measures: bool = False,
    ) -> pd.DataFrame:
        columns = {}
        for feat in mo.status.progress_bar(
            feature_list,
            title="Calculating Feature Agreement Scores",
            remove_on_exit=True,
        ):
            f_column = ("score", feat) if add_distance_measures else feat
            columns[f_column] = surveyAnswers.apply(
                lambda x, f=feat: getLocalFeatureAgreement(x, feature_df, f),
                axis=1,
            )

        if add_distance_measures:
            for feat in mo.status.progress_bar(
                feature_list,
                title="Calculating Feature Distances",
                remove_on_exit=True,
            ):
                if pd.api.types.is_float_dtype(feature_df[feat]):
                    continue
                columns[("distance", feat)] = surveyAnswers.apply(
                    lambda x, f=feat: getLocalFeatureAgreement(
                        x, feature_df, f, True
                    ),
                    axis=1,
                )

        agree_df = pd.DataFrame(columns)
        if add_distance_measures:
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

    return getAllAgreements, get_mean_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Set Agreement Scores

    - Earth Mover's Distance Algorithm from [GitHub](https://github.com/garydoranjr/pyemd)

    Doran, G. (2014). PyEMD: Earth Mover’s Distance for Python. https://github.com/garydoranjr/pyemd
    """)
    return


@app.cell
def _():
    from scipy.spatial.distance import euclidean

    return (euclidean,)


@app.cell
def _(Literal, euclidean, pd, surveyAnswers):
    def getDistanceAgreementRow(
        answer, feature_df, distance_algorithm
    ) -> pd.Series:
        x = feature_df.loc[answer["track_id_X"]].values
        a_1 = feature_df.loc[answer["track_id_1"]].values
        a_2 = feature_df.loc[answer["track_id_2"]].values
        if distance_algorithm == "euclidean":
            dist_a_1 = euclidean(x, a_1)
            dist_a_2 = euclidean(x, a_2)
        else:
            raise NotImplementedError(
                f"Distance Measure {distance_algorithm} is not implemented yet!"
            )

        dist = (dist_a_2 - dist_a_1) / (dist_a_1 + dist_a_2)
        acc_score = 1.0 if dist_a_1 < dist_a_2 else 0.0

        return pd.Series({"distance": dist, "accuracy": acc_score})


    def getGlobalDistanceAgreement(
        feature_df: pd.DataFrame,
        distance_algorithm: Literal["euclidean"] = "euclidean",
    ) -> pd.DataFrame:
        """
        Distance algorithms:
        - euclidean: use euclidean distance
        """
        gda_df = pd.DataFrame()
        gda_df[["distance", "accuracy"]] = surveyAnswers.apply(
            lambda x: getDistanceAgreementRow(x, feature_df, distance_algorithm),
            axis=1,
        )
        return gda_df

    return (getGlobalDistanceAgreement,)


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
def _(getAllAgreements, hl_features, track_df):
    hl_agreements = getAllAgreements(track_df, hl_features, True)
    hl_agreements
    return (hl_agreements,)


@app.cell
def _(get_mean_values, hl_agreements, sns):
    sns.barplot(
        y=hl_agreements["score"].columns,
        x=list(get_mean_values(hl_agreements["score"]).values()),
    )
    return


@app.cell
def _(get_mean_values, hl_agreements, sns):
    sns.barplot(
        y=hl_agreements["distance"].columns,
        x=list(get_mean_values(hl_agreements["distance"]).values()),
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
    import torchaudio
    from speechbrain.inference.encoders import MelSpectrogramEncoder
    from src.utils import get_trimmed_audio

    import torch

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
def _(encoder, get_embedding, np, pd, track_df):
    embedding_df = pd.DataFrame(
        np.stack(
            track_df.song_path.apply(lambda x: get_embedding(x, encoder)).values
        ),
        columns=[f"emb_{e}" for e in range(192)],
        index=track_df.index,
    )
    embedding_df
    return (embedding_df,)


@app.cell
def _(embedding_df, getGlobalDistanceAgreement):
    embedding_gda_df = getGlobalDistanceAgreement(
        embedding_df, distance_algorithm="euclidean"
    )
    embedding_gda_df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Fine Tuned Embedding Agreement
    """)
    return


@app.cell
def _(MODEL_FOLDER, encoder, os, torch):
    finetuned_encoder_path = os.path.join(
        MODEL_FOLDER,
        "finetuned_encoder",
        f"spkrec-ecapa-voxceleb-mel-spec_shuffle_32_0.3_40_1e-05.pt",
    )
    encoder.hparams.embedding_model.load_state_dict(
        torch.load(finetuned_encoder_path)
    )
    return


@app.cell
def _(encoder, get_embedding, np, pd, track_df):
    ft_embedding_df = pd.DataFrame(
        np.stack(
            track_df.song_path.apply(lambda x: get_embedding(x, encoder)).values
        ),
        columns=[f"emb_{e}" for e in range(192)],
        index=track_df.index,
    )
    ft_embedding_df
    return (ft_embedding_df,)


@app.cell
def _(ft_embedding_df, getGlobalDistanceAgreement):
    ft_embedding_gda_df = getGlobalDistanceAgreement(
        ft_embedding_df, distance_algorithm="euclidean"
    )
    ft_embedding_gda_df.describe()
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
        DATASET_FOLDER, "fma_large_feature_sets", "gemaps.npy"
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
def _(gemaps_feature_path, np, pd, smile_gemaps, track_df):
    gemaps_features_df = pd.DataFrame(
        np.load(gemaps_feature_path),
        columns=smile_gemaps.feature_names,
        index=track_df.index,
    )
    gemaps_features_df
    return (gemaps_features_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single Feature Agreement
    """)
    return


@app.cell
def _(gemaps_features_df, getAllAgreements):
    gemaps_agreements = getAllAgreements(
        gemaps_features_df, gemaps_features_df.columns
    )
    gemaps_agreements
    return (gemaps_agreements,)


@app.cell
def _(gemaps_agreements, get_mean_values, sns):
    top_gemaps_score_values = get_mean_values(gemaps_agreements, top_x=15)

    sns.barplot(
        y=list(top_gemaps_score_values.keys()),
        x=list(top_gemaps_score_values.values()),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature Set Agreement
    """)
    return


@app.cell
def _(gemaps_features_df, getGlobalDistanceAgreement):
    gemaps_gda_df = getGlobalDistanceAgreement(
        gemaps_features_df, distance_algorithm="euclidean"
    )
    gemaps_gda_df.describe()
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
        DATASET_FOLDER, "fma_large_feature_sets", "compare.npy"
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
def _(compare_feature_path, np, pd, smile_compare, track_df):
    compare_features_df = pd.DataFrame(
        np.load(compare_feature_path),
        columns=smile_compare.feature_names,
        index=track_df.index,
    )
    compare_features_df
    return (compare_features_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Single Feature Agreement
    """)
    return


@app.cell
def _(compare_features_df, getAllAgreements):
    compare_agreements = getAllAgreements(
        compare_features_df, compare_features_df.columns
    )
    compare_agreements
    return (compare_agreements,)


@app.cell
def _(compare_agreements, get_mean_values, sns):
    top_compare_score_values = get_mean_values(compare_agreements, top_x=15)

    sns.barplot(
        y=list(top_compare_score_values.keys()),
        x=list(top_compare_score_values.values()),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature Set Agreement
    """)
    return


@app.cell
def _(compare_features_df, getGlobalDistanceAgreement):
    compare_gda_df = getGlobalDistanceAgreement(
        compare_features_df, distance_algorithm="euclidean"
    )
    compare_gda_df.describe()
    return


if __name__ == "__main__":
    app.run()
