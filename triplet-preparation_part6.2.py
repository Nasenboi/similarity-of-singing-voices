import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import librosa
    import marimo as mo
    import numpy
    import numpy as np
    import pandas as pd
    import torch
    from dotenv import load_dotenv
    from safetensors.torch import load_file, save_file
    from transformers import AutoModel, AutoProcessor, pipeline

    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    load_dotenv(os.path.join(BASEDIR, ".env"))
    return (
        AutoModel,
        AutoProcessor,
        librosa,
        load_file,
        mo,
        np,
        os,
        pd,
        save_file,
        torch,
    )


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
    return (
        CSV_FOLDER,
        EMBEDDING_FOLDER,
        MODEL_FOLDER,
        RANDOM_STATE,
        RECALC_EMBEDDINGS,
        SAMPLE_RATE,
    )


@app.cell
def _(CSV_FOLDER, os, pd):
    triplet_df = pd.read_csv(
        os.path.join(CSV_FOLDER, "triplet_df_checked_all.csv"),
        index_col="track_id",
    )
    triplet_df = triplet_df[triplet_df.is_voiced]
    triplet_df
    return (triplet_df,)


@app.cell
def _(SAMPLE_RATE, librosa, np):
    def getTrimmedAudio(audiopath: str):
        y, sr = librosa.load(audiopath, sr=SAMPLE_RATE, mono=True)
        intervals = librosa.effects.split(y)
        trimmed_parts = []
        for start, end in intervals:
            trimmed_parts.append(y[..., start:end])
        return np.concatenate(trimmed_parts, axis=-1)
    return (getTrimmedAudio,)


@app.cell
def _(AutoModel, AutoProcessor, MODEL_FOLDER):
    processor = AutoProcessor.from_pretrained(
        "marksverdhei/Qwen3-Voice-Embedding-12Hz-0.6B",
        trust_remote_code=True,
        cache_dir=MODEL_FOLDER,
    )
    model = AutoModel.from_pretrained(
        "marksverdhei/Qwen3-Voice-Embedding-12Hz-0.6B",
        trust_remote_code=True,
        cache_dir=MODEL_FOLDER,
    )
    model.eval()
    return model, processor


@app.cell
def _(SAMPLE_RATE, getTrimmedAudio, model, processor, triplet_df):
    def getEmbedding(vocal_audio_path: str):
        y = getTrimmedAudio(triplet_df.iloc[0].vocal_audio_path)
        inputs = processor(y, sampling_rate=SAMPLE_RATE)
        embedding = model(**inputs).last_hidden_state
        return embedding.squeeze()
    return (getEmbedding,)


@app.cell
def _():
    return


@app.cell
def _(
    EMBEDDING_FOLDER,
    RECALC_EMBEDDINGS,
    getEmbedding,
    load_file,
    np,
    os,
    save_file,
    torch,
    triplet_df,
):
    if RECALC_EMBEDDINGS:
        with torch.no_grad():
            embeddings = np.vstack(triplet_df.vocal_audio_path.apply(getEmbedding))
        embeddings_tensor = torch.from_numpy(embeddings)
        save_file(
            {"voice_embeddings": embeddings_tensor},
            os.path.join(EMBEDDING_FOLDER, "triplet_no_finetune.safetensors"),
        )
    else:
        loaded = load_file(
            os.path.join(EMBEDDING_FOLDER, "triplet_no_finetune.safetensors")
        )
        embeddings = loaded["voice_embeddings"].numpy()
    return (embeddings,)


@app.cell
def _(embeddings, pd, triplet_df):
    embedding_df = pd.DataFrame(embeddings, index=triplet_df.index)
    return (embedding_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # UMAP Dim Reduction
    """)
    return


@app.cell
def _():
    import umap
    from sklearn.preprocessing import StandardScaler
    return StandardScaler, umap


@app.cell
def _(StandardScaler, embeddings):
    scaler = StandardScaler()
    voice_features_scaled = scaler.fit_transform(embeddings)
    return (voice_features_scaled,)


@app.cell
def _(RANDOM_STATE, pd, triplet_df, umap, voice_features_scaled):
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
    return (umap_df,)


@app.cell
def _(triplet_df, umap_df):
    triplet_df_umap = triplet_df.join(umap_df)
    triplet_df_umap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Validation of Data
    """)
    return


@app.cell
def _(embedding_df, np, triplet_df):
    def getArtistRange(artist: str):
        artist_embeddings = embedding_df[triplet_df.artist == artist]
        # center_point = artist_embeddings.mean(axis=0).values

        max_vals = artist_embeddings.max(axis=0).values
        min_vals = artist_embeddings.min(axis=0).values

        return np.mean(max_vals - min_vals)
    return (getArtistRange,)


@app.cell
def _(getArtistRange, pd, triplet_df):
    artists = triplet_df.artist.unique()
    artist_range = [getArtistRange(a) for a in artists]
    artist_df = pd.DataFrame({"artist": artists}) #"range": artist_range
    artist_df
    return (artist_df,)


@app.cell
def _(embedding_df, triplet_df):
    embedding_df[triplet_df.artist == "Mount Eerie"]
    return


@app.cell
def _(artist_df, embedding_df, triplet_df):
    artist_df["center_point"] =  artist_df.artist.apply(lambda a: embedding_df[triplet_df.artist == a].mean(axis=0).values)
    artist_df
    return


@app.cell
def _(artist_df):
    artist_df.iloc[0].center_point
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
