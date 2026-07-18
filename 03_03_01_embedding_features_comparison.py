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
    import pandas as pd
    import seaborn as sns
    import torch
    import torchaudio
    from sklearn.preprocessing import StandardScaler
    from speechbrain.inference.encoders import MelSpectrogramEncoder

    from src import load_singer_identity_model
    from src.globals import (AUDIO_FOLDER, CSV_FOLDER, DATASET_FOLDER,
                             MODEL_FOLDER, PLOT_FOLDER, STEMS_FOLDER,
                             TRACKS_PATH, UVR_MODEL_PATH)
    from src.statistics.feature_correlation import (
        get_all_distance_differences, get_global_distance_scores, scale_df)
    from src.statistics.plotting import (plot_correlation_bar,
                                         plot_correlation_scatter)
    from src.survey_dataset_helpers import load_survey_data
    from src.utils import get_trimmed_audio

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        MODEL_FOLDER,
        MelSpectrogramEncoder,
        PLOT_FOLDER,
        get_global_distance_scores,
        get_trimmed_audio,
        load_singer_identity_model,
        load_survey_data,
        mo,
        np,
        os,
        pd,
        plot_correlation_bar,
        plot_correlation_scatter,
        scale_df,
        torch,
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
def _(MelSpectrogramEncoder):
    encoder = MelSpectrogramEncoder.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb-mel-spec")
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
        np.stack(track_df.song_path.apply(lambda x: get_embedding(x, encoder)).values),
        columns=[f"emb_{e}" for e in range(192)],
        index=track_df.index,
    )
    embedding_df = scale_df(embedding_df)
    embedding_df
    return (embedding_df,)


@app.cell
def _(embedding_df, get_global_distance_scores, questions_df):
    embedding_gda_df = get_global_distance_scores(embedding_df, questions_df)
    embedding_gda_df
    return (embedding_gda_df,)


@app.cell
def _(embedding_gda_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="ECAPA-TDNN Embeddings Correlations (Randomized)",
        feature_df=embedding_gda_df[questions_df.randomized],
        target_feature=questions_df[questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(embedding_gda_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="ECAPA-TDNN Embeddings Correlations (Max Entropy)",
        feature_df=embedding_gda_df[~questions_df.randomized],
        target_feature=questions_df[~questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(embedding_gda_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="ECAPA-TDNN Embeddings Correlations (All)",
        feature_df=embedding_gda_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, embedding_gda_df, plot_correlation_scatter, questions_df):
    plot_correlation_scatter(
        title="ECAPA-TDNN Embeddings (Cosine Distance)",
        feature_name="ECAPA_Embeddings_Cosine",
        y=embedding_gda_df["distance_cosine"],
        x=questions_df["A_perc"],
        plot_dir=PLOT_SAVE_DIR,
    )
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
    sID_SAMPLE_RATE = 44100
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
            np.stack(track_df.song_path.apply(lambda x: get_singer_ID_embedding(x, singer_ID_model)).values),
            columns=[f"emb_{e}" for e in range(1000)],
            index=track_df.index,
        )
    sID_embedding_df = scale_df(sID_embedding_df)
    sID_embedding_df
    return (sID_embedding_df,)


@app.cell
def _(get_global_distance_scores, questions_df, sID_embedding_df):
    sID_embedding_gda_df = get_global_distance_scores(sID_embedding_df, questions_df)
    return (sID_embedding_gda_df,)


@app.cell
def _(plot_correlation_bar, questions_df, sID_embedding_gda_df):
    plot_correlation_bar(
        title="Singer ID Embeddings Correlations (Randomized)",
        feature_df=sID_embedding_gda_df[questions_df.randomized],
        target_feature=questions_df[questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(plot_correlation_bar, questions_df, sID_embedding_gda_df):
    plot_correlation_bar(
        title="Singer ID Embeddings Correlations (Max Entropy)",
        feature_df=sID_embedding_gda_df[~questions_df.randomized],
        target_feature=questions_df[~questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(plot_correlation_bar, questions_df, sID_embedding_gda_df):
    plot_correlation_bar(
        title="Singer ID Embeddings Correlations (All)",
        feature_df=sID_embedding_gda_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    plot_correlation_scatter,
    questions_df,
    sID_embedding_gda_df,
):
    plot_correlation_scatter(
        title="Singer ID Embeddings (Cosine Distance)",
        feature_name="Singer_ID_Embeddings_Cosine",
        y=sID_embedding_gda_df["distance_cosine"],
        x=questions_df["A_perc"],
        plot_dir=PLOT_SAVE_DIR,
    )
    return


if __name__ == "__main__":
    app.run()
