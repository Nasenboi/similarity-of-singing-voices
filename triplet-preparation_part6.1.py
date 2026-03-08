import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy
    import numpy
    import os
    from dotenv import load_dotenv
    import librosa
    import numpy as np
    from transformers import pipeline
    import torch
    from transformers import AutoModel, AutoProcessor
    from safetensors.torch import save_file, load_file
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.dataio import audio_io

    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    load_dotenv(os.path.join(BASEDIR, ".env"))
    return SpeakerRecognition, librosa, mo, np, os, pd, torch


@app.cell
def _(os):
    RANDOM_STATE = 42
    RECALC_EMBEDDINGS = False
    DATASET_FOLDER = os.getenv("DATASET_FOLDER")
    CSV_FOLDER = os.getenv("CSV_FOLDER")
    MODEL_FOLDER = os.getenv("MODEL_FOLDER")
    AUDIO_FOLDER = os.path.join(DATASET_FOLDER, "fma_large")
    STEM_FOLDER = os.path.join(DATASET_FOLDER, "fma_large_stems")
    EMBEDDING_FOLDER = os.path.join(DATASET_FOLDER, "fma_large_embeddings")
    VOCALS_FILE_NAME = "vocals.mp3"
    SAMPLE_RATE = 24_000
    return CSV_FOLDER, RANDOM_STATE, SAMPLE_RATE


@app.cell
def _(CSV_FOLDER, os, pd):
    triplet_df = pd.read_csv(
        os.path.join(CSV_FOLDER, "triplet_df_checked_all.csv"),
        index_col="track_id",
    )
    triplet_df = triplet_df[triplet_df.is_voiced]
    triplet_df
    return (triplet_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Speechbrain
    """)
    return


@app.cell
def _(SAMPLE_RATE, librosa, np, torch):
    def getTrimmedAudio(audiopath: str):
        y, sr = librosa.load(audiopath, sr=SAMPLE_RATE, mono=True)
        intervals = librosa.effects.split(y)
        trimmed_parts = []
        for start, end in intervals:
            trimmed_parts.append(y[..., start:end])
        return torch.from_numpy(np.concatenate(trimmed_parts, axis=-1))
    return (getTrimmedAudio,)


@app.cell
def _(SpeakerRecognition):
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
    )
    return (verification,)


@app.cell
def _():
    a1 = 10
    a2 = 140
    return a1, a2


@app.cell
def _(a1, a2, getTrimmedAudio, triplet_df, verification):
    verification.verify_batch(
        getTrimmedAudio(triplet_df.loc[a1].vocal_audio_path),
        getTrimmedAudio(triplet_df.loc[a2].vocal_audio_path)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Speechbrain embedding
    """)
    return


@app.cell
def _():
    import torchaudio
    from speechbrain.inference.encoders import MelSpectrogramEncoder

    encoder = MelSpectrogramEncoder.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb-mel-spec")
    return (encoder,)


@app.cell
def _(encoder, getTrimmedAudio, np, triplet_df):
    embeddings = np.stack(triplet_df.vocal_audio_path.apply(
        lambda path: encoder.encode_waveform(getTrimmedAudio(path)).cpu().numpy().squeeze()
    ).values)
    return (embeddings,)


@app.cell
def _(embeddings, pd, triplet_df):
    embeddings_df = pd.DataFrame(embeddings, index=triplet_df.index)
    embeddings_df
    return (embeddings_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Umap
    """)
    return


@app.cell
def _(RANDOM_STATE, embeddings, pd, triplet_df):
    import umap
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    voice_features_scaled = scaler.fit_transform(embeddings)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    umap_embeddings = reducer.fit_transform(voice_features_scaled)

    umap_df = pd.DataFrame(
        umap_embeddings, index=triplet_df.index, columns=["UMAP1", "UMAP2"]
    )

    triplet_df_umap = triplet_df.join(umap_df)
    triplet_df_umap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification
    """)
    return


@app.cell
def _(embeddings_df, np, triplet_df):
    def getArtistRange(artist: str):
        artist_embeddings = embeddings_df[triplet_df.artist == artist]
        # center_point = artist_embeddings.mean(axis=0).values

        max_vals = artist_embeddings.max(axis=0).values
        min_vals = artist_embeddings.min(axis=0).values

        return np.mean(max_vals - min_vals)
    return (getArtistRange,)


@app.cell
def _(getArtistRange, pd, triplet_df):
    artists = triplet_df.artist.unique()
    artist_range = [getArtistRange(a) for a in artists]
    artist_df = pd.DataFrame({"artist": artists, "range": artist_range}) #
    artist_df
    return (artist_df,)


@app.cell
def _(artist_df, embeddings_df, triplet_df):
    artist_df["center_point"] =  artist_df.artist.apply(lambda a: embeddings_df[triplet_df.artist == a].mean(axis=0).values)
    return


@app.cell
def _(artist_df, np):
    from scipy.spatial.distance import pdist, squareform

    points = np.stack(artist_df['center_point'].values)
    distance_matrix = squareform(pdist(points))
    return (distance_matrix,)


@app.cell
def _(artist_df, distance_matrix, np, pd):
    flat_indices = np.argsort(distance_matrix, axis=None)[:10]
    i_indices, j_indices = np.unravel_index(flat_indices, distance_matrix.shape)

    top_pairs = []
    for i, j in zip(i_indices, j_indices):
        top_pairs.append({
            'artist1': artist_df.iloc[i]['artist'],
            'artist2': artist_df.iloc[j]['artist'],
            'distance': distance_matrix[i, j]
        })

    flat_indices = np.argsort(distance_matrix, axis=None)[-10:][::-1]
    i_indices, j_indices = np.unravel_index(flat_indices, distance_matrix.shape)
    most_different_pairs = []
    for i, j in zip(i_indices, j_indices):
        most_different_pairs.append({
            'artist1': artist_df.iloc[i]['artist'],
            'artist2': artist_df.iloc[j]['artist'],
            'distance': distance_matrix[i, j]
        })
    most_different_10_df = pd.DataFrame(most_different_pairs)
    top_10_df = pd.DataFrame(top_pairs)
    top_10_df
    return (most_different_10_df,)


@app.cell
def _(most_different_10_df):
    most_different_10_df
    return


if __name__ == "__main__":
    app.run()
