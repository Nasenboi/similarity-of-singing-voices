import marimo

__generated_with = "0.23.13"
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
    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    import random

    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tensorflow as tf

    # from src import load_singer_identity_model

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

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        PLOT_FOLDER,
        get_global_distance_scores,
        librosa,
        load_survey_data,
        mo,
        np,
        os,
        pd,
        plot_correlation_bar,
        plot_correlation_scatter,
        random,
        scale_df,
        tf,
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
def _(np, random, tf):
    RANDOM_SEED = 72
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    return


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
def _(PLOT_FOLDER, os, plot_correlation_scatter, questions_df):
    PLOT_SAVE_DIR = os.path.join(PLOT_FOLDER, "survey_2")

    def plot_feature_correlation_scatter(
        feature_name: str, feature, target_feature=questions_df["A_perc"]
    ):
        plot_correlation_scatter(
            title=f"{feature_name} Feature Correlation",
            x=target_feature,
            y=feature,
            save_path=os.path.join(
                PLOT_SAVE_DIR, f"questions_{feature_name}_correlation.png"
            ),
            legend_loc="lower right",
        )

    return PLOT_SAVE_DIR, plot_feature_correlation_scatter


@app.cell
def _(mo):
    mo.md(r"""
    # CVSM Embedding Egreement Agreement
    Calculate the embeddings using the previously used embedding model:
    - CVSM ART

    Garoufis, C., Zlatintsi, A., & Maragos, P. (2025). CVSM: Contrastive Vocal Similarity Modeling. arXiv [Eess.AS]. Retrieved from http://arxiv.org/abs/2510.03025
    """)
    return


@app.cell
def _():
    from src.submodules.cvsm.cola import constants
    from src.submodules.cvsm.mscol import network

    return constants, network


@app.cell
def _(os):
    cvsm_model_path = os.path.join(
        os.path.dirname(__file__),
        "src",
        "submodules",
        "cvsm",
        "models",
        "htdemucs",
        "m4all_cvsmart",
    )
    return (cvsm_model_path,)


@app.cell
def _(constants, cvsm_model_path, network, tf):
    contrastive_network = network.get_contrastive_network(
        embedding_dim=512,
        temperature=0.2,
        pooling_type="max",
        similarity_type=constants.SimilarityMeasure.BILINEAR,
    )
    contrastive_network.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    contrastive_network.load_weights(
        tf.train.latest_checkpoint(cvsm_model_path)
    ).expect_partial()
    encoder = contrastive_network.embedding_model.get_layer("encoder")
    return (encoder,)


@app.cell
def _(encoder, tf):
    inputs = tf.keras.layers.Input(shape=(16000,))
    x = tf.signal.stft(
        inputs, frame_length=400, frame_step=160, fft_length=1024
    )
    x = tf.abs(x)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        64, x.shape[-1], 16000, 60, 7800
    )
    x = tf.tensordot(x, mel_matrix, 1)
    x = tf.clip_by_value(x, clip_value_min=1e-5, clip_value_max=1e8)
    x = tf.expand_dims(tf.math.log(x), axis=-1)
    outputs = encoder(x)
    model = tf.keras.Model(inputs, outputs)
    return (model,)


@app.cell
def _(librosa, model, np, tf):
    CHUNK_SIZE = 16000
    SAMPLE_RATE = 16000

    def get_embedding(song_path, discard_silent=False, silence_thresh=0.01):
        y, _ = librosa.load(song_path, sr=SAMPLE_RATE, mono=True)

        n_chunks = len(y) // CHUNK_SIZE
        if n_chunks == 0:
            raise ValueError(f"Audio too short: {song_path}")
        y = y[: n_chunks * CHUNK_SIZE]

        data = np.reshape(y, (n_chunks, CHUNK_SIZE)).astype(np.float32)

        if discard_silent:
            mean_amp = np.mean(np.abs(data), axis=1)
            data = data[mean_amp >= silence_thresh]
            if len(data) == 0:
                raise ValueError(f"All chunks silent: {song_path}")

        embs = model.predict(data, verbose=0)
        embedding = tf.math.l2_normalize(embs.mean(axis=0))  # (1280,)

        return embedding.numpy()

    return (get_embedding,)


@app.cell
def _(encoder, get_embedding, np, pd, scale_df, track_df):
    embedding_df = pd.DataFrame(
        np.stack(
            track_df.vocal_path.apply(
                lambda x: get_embedding(x, encoder)
            ).values
        ),
        columns=[f"emb_{e}" for e in range(1280)],
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
        title="CVSM Embeddings Correlations (Randomized)",
        feature_df=embedding_gda_df[questions_df.randomized],
        target_feature=questions_df[questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(embedding_gda_df, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="CVSM Embeddings Correlations (Max Entropy)",
        feature_df=embedding_gda_df[~questions_df.randomized],
        target_feature=questions_df[~questions_df.randomized]["A_perc"],
        top_x=10,
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, embedding_gda_df, os, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="CVSM Embeddings Correlations (All)",
        feature_df=embedding_gda_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
        save_path=os.path.join(
            PLOT_SAVE_DIR, "CVSM Embeddings Correlations (All).png"
        ),
    )
    return


@app.cell
def _(embedding_gda_df, plot_feature_correlation_scatter):
    plot_feature_correlation_scatter(
        "CVSM Embeddings (Canberra Distance)",
        embedding_gda_df["distance_canberra"],
    )
    return


if __name__ == "__main__":
    app.run()
