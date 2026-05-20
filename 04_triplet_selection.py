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
    # Initial Imports
    import marimo as mo
    import pandas as pd
    import numpy as np
    import torch
    import os

    from src.globals import CSV_FOLDER, DATASET_FOLDER
    from src.utils import get_trimmed_audio
    return CSV_FOLDER, DATASET_FOLDER, get_trimmed_audio, mo, np, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load the Dataset
    """)
    return


@app.cell
def _(CSV_FOLDER, os, pd):
    subpath = "LargeDataset/triplet_selection"
    filename = "dataset_vq3.csv"
    df = pd.read_csv(
        os.path.join(CSV_FOLDER, subpath, filename), index_col="track_id"
    )
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    # Set Paths and Hyperparameters
    """)
    return


@app.cell
def _(DATASET_FOLDER, os):
    embeddings_path = os.path.join(
        DATASET_FOLDER, "fma_large_embeddings", f"mel_spec_enc_nlognK.safetensors"
    )
    triplets_path = os.path.join(
        DATASET_FOLDER, "fma_large_triplets", f"mel_spec_enc_nlognK.npy"
    )
    return embeddings_path, triplets_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Audio Files
    """)
    return


@app.cell
def _(df, embeddings_path, get_trimmed_audio, os):
    if not os.path.exists(embeddings_path):
        X = df.vocal_path.apply(get_trimmed_audio)
        print(X.shape)
    else:
        X = None
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Embedding Model
    - MelSpectrogramEncoder from [pip](https://speechbrain.readthedocs.io/en/latest/installation.html), thanks to speechbrain
    - Encoder Model from [HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb-mel-spec), thanks to speechbrain

    Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., Subakan, C., Dawalatabad, N., Heba, A., Zhong, J., Chou, J.-C., Yeh, S.-L., Fu, S.-W., Liao, C.-F., Rastorgueva, E., Grondin, F., Aris, W., Na, H., Gao, Y., … Bengio, Y. (2021). SpeechBrain: A General-Purpose Speech Toolkit.
    """)
    return


@app.cell
def _():
    import torchaudio
    from speechbrain.inference.encoders import MelSpectrogramEncoder

    embedder = MelSpectrogramEncoder.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb-mel-spec"
    )
    return (embedder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Apply Triplet Selection
    - MaxEntTripletSelector from [GitHub](https://github.com/Nasenboi/torch-maxent-triplet-selection), self written, based on work of Priyadarshini et. al. (2021)

    K, Priyadarshini, Chaudhuri, S., Borkar, V., & Chaudhuri, S. (2021). A unified batch selection policy for active metric learning. [doi:10.48550/ARXIV.2102.07365](https://arxiv.org/abs/2102.07365)
    """)
    return


@app.cell
def _(X, df, embedder, embeddings_path, triplets_path):
    from maxent_triplet_selector import MaxEntTripletSelector

    selector = MaxEntTripletSelector(embedder=embedder)
    best_batch = selector.get_best_batch(
        X=X, embeddings_path=embeddings_path, df=df, batch_path=triplets_path
    )
    best_batch
    return best_batch, selector


@app.cell
def _(X, df, embeddings_path, selector, triplets_path):
    selector.get_best_batch(
        X=X, embeddings_path=embeddings_path, df=df, batch_path=triplets_path, use_index_col=False
    )
    return


@app.cell
def _(best_batch, np, triplets_path):
    with open(triplets_path.replace(".npy", "_index.npy"), "wb") as f:
        np.save(f, best_batch.astype(np.int32))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Add additional Metadata to the DataFrame
    1. The cluster ids
    2. A dimensionality reduced embedding Version for a 2D / 3D map
    """)
    return


@app.cell
def _(df, selector):
    df["cluster"] = selector.cluster_ids
    df
    return


@app.cell
def _(selector):
    mean_embeddings = selector.embeddings.mean(dim=1)
    mean_embeddings.shape
    return (mean_embeddings,)


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    return (StandardScaler,)


@app.cell
def _(StandardScaler, mean_embeddings):
    scaler = StandardScaler()

    mean_embeddings_scaled = scaler.fit_transform(mean_embeddings)
    mean_embeddings_scaled.shape
    return (mean_embeddings_scaled,)


@app.cell
def _():
    from umap import UMAP
    return (UMAP,)


@app.cell
def _(UMAP):
    umap_settings = {"n_neighbors": 5, "min_dist": 0.1, "metric": "cosine"}

    reducer_2d = UMAP(n_components=2, **umap_settings)
    return (umap_settings,)


@app.cell
def _(UMAP, df, mean_embeddings_scaled, pd, umap_settings):
    df_joined = df
    for dim in [2, 3]:
        reducer = UMAP(
            n_components=dim, **umap_settings
        )
        umap_embeddings = reducer.fit_transform(mean_embeddings_scaled)

        embedding_df = pd.DataFrame(
            umap_embeddings, index=df.index, columns=[f"UMAP_{dim}D_{d+1}" for d in range(dim)]
        )
        df_joined = df_joined.join(embedding_df)

    df_joined
    return


if __name__ == "__main__":
    app.run()
