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

    from src.globals import (
        CSV_FOLDER,
        DATASET_FOLDER,
        TRACKS_PATH,
        AUDIO_FOLDER,
        STEMS_FOLDER,
        UVR_MODEL_PATH,
        MODEL_FOLDER,
    )
    return CSV_FOLDER, DATASET_FOLDER, MODEL_FOLDER, mo, os, pd, sns


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
def _(hl_features, track_df):
    track_df[hl_features]
    return


@app.cell
def _(pd, track_df):
    def getAgreementScore(x, a_1, a_2):
        if pd.api.types.is_numeric_dtype(type(x)):
            dist_a_1 = abs(x - a_1)
            dist_a_2 = abs(x - a_2)
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


    def getFeatureAgreement(answer, feature: str):
        x = track_df.loc[answer["track_id_X"]][feature]
        a_1 = track_df.loc[answer["track_id_1"]][feature]
        a_2 = track_df.loc[answer["track_id_2"]][feature]
        return getAgreementScore(x, a_1, a_2)
    return (getFeatureAgreement,)


@app.cell
def _(getFeatureAgreement, hl_features, mo, surveyAnswers):
    for feat in mo.status.progress_bar(hl_features):
        surveyAnswers[f"{feat}_agreement"] = surveyAnswers.apply(
            lambda x: getFeatureAgreement(x, feat), axis=1
        )
    surveyAnswers
    return


@app.cell
def _(hl_features, surveyAnswers):
    agreement_mean_values = {
        feat: surveyAnswers[f"{feat}_agreement"].mean() for feat in hl_features
    }
    return (agreement_mean_values,)


@app.cell
def _(agreement_mean_values, sns):
    sns.barplot(
        y=list(agreement_mean_values.keys()),
        x=list(agreement_mean_values.values()),
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
    from scipy.spatial.distance import euclidean
    import torch
    return MelSpectrogramEncoder, euclidean, get_trimmed_audio, torch


@app.cell
def _(MelSpectrogramEncoder):
    encoder = MelSpectrogramEncoder.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb-mel-spec"
    )
    return (encoder,)


@app.cell
def _(encoder, get_trimmed_audio, track_df):
    embeddings = track_df.song_path.apply(
        lambda x: encoder.encode_waveform(get_trimmed_audio(x))
        .cpu()
        .numpy()
        .squeeze()
    )
    embeddings.shape
    return (embeddings,)


@app.cell
def _(embeddings, euclidean, surveyAnswers):
    def getEmbeddingAgreementScore(answer, emb):
        x = emb.loc[answer["track_id_X"]]
        a_1 = emb.loc[answer["track_id_1"]]
        a_2 = emb.loc[answer["track_id_2"]]
        dist_a_1 = euclidean(x, a_1)
        dist_a_2 = euclidean(x, a_2)
        if dist_a_1 < dist_a_2:
            return 1.0  # => agreement
        if dist_a_2 < dist_a_1:
            return -1.0  # => disagreement
        else:
            return 0.0  # => uncertainty


    surveyAnswers["embedding_agreement"] = surveyAnswers.apply(
        lambda x: getEmbeddingAgreementScore(x, embeddings), axis=1
    )
    surveyAnswers["embedding_agreement"].mean()
    return (getEmbeddingAgreementScore,)


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
def _(encoder, get_trimmed_audio, track_df):
    ft_embeddings = track_df.song_path.apply(
        lambda x: encoder.encode_waveform(get_trimmed_audio(x))
        .cpu()
        .numpy()
        .squeeze()
    )
    ft_embeddings.shape
    return (ft_embeddings,)


@app.cell
def _(ft_embeddings, getEmbeddingAgreementScore, surveyAnswers):
    surveyAnswers["ft_embedding_agreement"] = surveyAnswers.apply(
        lambda x: getEmbeddingAgreementScore(x, ft_embeddings), axis=1
    )
    surveyAnswers["ft_embedding_agreement"].mean()
    return


if __name__ == "__main__":
    app.run()
