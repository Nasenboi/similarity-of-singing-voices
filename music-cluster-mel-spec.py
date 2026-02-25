import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from os import getenv, path
    from random import choice

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    import utils

    load_dotenv()
    return choice, getenv, librosa, mo, np, path, utils


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create a Cluster of Musical Tracks
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Initialize Project: Import libs and set globals
    """)
    return


@app.cell
def _(getenv, path):
    RANDOM_STATE = 42
    DATASET_FOLDER = getenv("DATASET_FOLDER")
    METADATA_FOLDER = path.join(DATASET_FOLDER, "fma_metadata")
    AUDIO_FOLDER = path.join(DATASET_FOLDER, "fma_large")
    TRACKS_PATH = path.join(METADATA_FOLDER, "tracks.csv")
    # GENRES_PATH = path.join(METADATA_FOLDER, "genres.csv")
    # FEATURES_PATH = path.join(METADATA_FOLDER, "features.csv")
    return AUDIO_FOLDER, RANDOM_STATE, TRACKS_PATH


@app.cell
def _(TRACKS_PATH, utils):
    tracks_df = utils.load(TRACKS_PATH)
    tracks_df = tracks_df[tracks_df["set", "subset"] <= "small"]
    return (tracks_df,)


@app.cell
def _(tracks_df):
    tracks_df
    return


@app.cell
def _(tracks_df):
    tracks_df["track", "genre_top"]
    return


@app.cell
def _(AUDIO_FOLDER, tracks_df, utils):
    tracks_df["audio_path"] = [
        utils.get_audio_path(AUDIO_FOLDER, i) for i in tracks_df.index
    ]
    return


@app.cell
def _():
    # sum([path.exists(p) for p in tracks_df["audio_path"]] * 1)
    return


@app.cell
def _(choice, mo, tracks_df):
    audio_filename = choice(tracks_df["audio_path"].tolist())
    mo.audio(audio_filename)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create Features for the Cluster
    """)
    return


@app.cell
def _():
    SAMPLE_RATE = 16_000
    MAX_LEN_S = 10
    MAX_N = SAMPLE_RATE * MAX_LEN_S
    HOP_LENGTH = 512
    N_MELS = 64
    return HOP_LENGTH, MAX_LEN_S, MAX_N, N_MELS, SAMPLE_RATE


@app.cell
def _(HOP_LENGTH, MAX_LEN_S, MAX_N, N_MELS, SAMPLE_RATE, librosa, np):
    def generate_spec(path: str, flatten: bool = False) -> np.ndarray:
        try:
            y, sr = librosa.load(
                path, mono=True, sr=SAMPLE_RATE, duration=MAX_LEN_S
            )
            if len(y) < MAX_N:
                y = np.pad(y, (0, MAX_N - len(y)))
            else:
                y = y[:MAX_N]
            S = librosa.feature.melspectrogram(
                y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH
            )
            if not flatten:
                return S
            return S.flatten()
        except Exception as e:
            # print(e)
            return None
    return (generate_spec,)


@app.cell
def _():
    """
    S = generate_spec(audio_filename)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel", sr=SAMPLE_RATE, ax=ax
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel-frequency spectrogram")
    """
    return


@app.cell
def _():
    """
    S_flat = generate_spec(audio_filename, flatten=True)
    S_flat
    """
    return


@app.cell
def _(tracks_df):
    train_df = tracks_df[tracks_df["set", "split"] == "training"].copy()
    # vaildation_df = tracks_df[tracks_df["set", "split"] == "validation"]
    # test_df = tracks_df[tracks_df["set", "split"] == "test"]
    return (train_df,)


@app.cell
def _(generate_spec, train_df):
    train_spec_list = train_df["audio_path"].apply(
        lambda x: generate_spec(x, flatten=True)
    )

    none_positions = [i for i, spec in enumerate(train_spec_list) if spec is None]
    none_indices = train_df.index[none_positions]
    train_spec_list = [spec for spec in train_spec_list if spec is not None]
    train_df.drop(none_indices, inplace=True)
    return (train_spec_list,)


@app.cell
def _(np, train_spec_list):
    train_specs = np.array(train_spec_list)
    return (train_specs,)


@app.cell
def _(train_specs):
    from sklearn.preprocessing import StandardScaler

    spec_scaler = StandardScaler()
    train_specs_scaled = spec_scaler.fit_transform(train_specs)
    return (train_specs_scaled,)


@app.cell
def _(RANDOM_STATE, train_specs_scaled):
    import umap

    cluster_model = umap.UMAP(
        n_neighbors=5, n_components=2, random_state=RANDOM_STATE
    )

    embeddings = cluster_model.fit_transform(train_specs_scaled)
    return (embeddings,)


@app.cell
def _(embeddings, train_df):
    train_df["UMAP_1"] = embeddings[:, 0]
    train_df["UMAP_2"] = embeddings[:, 1]
    return


@app.cell
def _(train_df):
    train_df
    return


if __name__ == "__main__":
    app.run()
