import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import gc
    import os
    from itertools import permutations

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import torch
    from dotenv import load_dotenv
    from safetensors.torch import load_file, save_file
    from scipy.cluster.hierarchy import linkage
    from scipy_cut_tree_balanced import cut_tree_balanced
    from speechbrain.dataio import audio_io
    from speechbrain.inference.speaker import SpeakerRecognition
    from transformers import AutoModel, AutoProcessor, pipeline

    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    load_dotenv(os.path.join(BASEDIR, ".env"))
    return (
        cut_tree_balanced,
        gc,
        librosa,
        linkage,
        mo,
        np,
        os,
        pd,
        permutations,
        px,
        torch,
    )


@app.cell
def _(np, os):
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    RECALC_EMBEDDINGS = False
    DATASET_FOLDER = os.getenv("DATASET_FOLDER")
    CSV_FOLDER = os.getenv("CSV_FOLDER")
    MODEL_FOLDER = os.getenv("MODEL_FOLDER")
    AUDIO_FOLDER = os.path.join(DATASET_FOLDER, "fma_large")
    STEM_FOLDER = os.path.join(DATASET_FOLDER, "fma_large_stems")
    EMBEDDING_FOLDER = os.path.join(DATASET_FOLDER, "fma_large_embeddings")
    VOCALS_FILE_NAME = "vocals.mp3"
    batches_path = os.path.join(CSV_FOLDER, "SpeechBrain", f"batches.npy")
    SAMPLE_RATE = 24_000
    RECALC_EMB = False
    K = 300
    G = 10
    return (
        AUDIO_FOLDER,
        CSV_FOLDER,
        G,
        K,
        RECALC_EMB,
        SAMPLE_RATE,
        STEM_FOLDER,
        batches_path,
    )


@app.cell
def _():
    """
    triplet_df = pd.read_csv(
        os.path.join(CSV_FOLDER, "triplet_df_checked_all.csv"),
        index_col="track_id",
    )
    triplet_df = triplet_df[triplet_df.is_voiced]

    triplet_df
    """
    return


@app.cell
def _(SAMPLE_RATE, librosa, np, torch):
    def getTrimmedAudio(audiopath: str, use_torch=True):
        y, sr = librosa.load(audiopath, sr=SAMPLE_RATE, mono=True)
        intervals = librosa.effects.split(y)
        trimmed_parts = []
        for start, end in intervals:
            trimmed_parts.append(y[..., start:end])
        ta = np.concatenate(trimmed_parts, axis=-1)
        if not use_torch:
            return ta
        return torch.from_numpy(ta)
    return (getTrimmedAudio,)


@app.cell
def _(CSV_FOLDER, os, pd):
    """
    artists = triplet_df.artist.unique()

    single_artist_rows = []
    for artist in artists:
        artist_rows = triplet_df[triplet_df.artist == artist]
        if len(artist_rows) == 1:
            single_artist_rows.append(artist_rows.index[0])
            continue

        max_len_idx = None
        max_len = 0
        for idx, row in artist_rows.iterrows():
            row_len = len(getTrimmedAudio(row.vocal_audio_path, use_torch=False))
            if row_len > max_len:
                max_len = row_len
                max_len_idx = idx
        single_artist_rows.append(max_len_idx)

    single_artist_df = triplet_df.loc[single_artist_rows]
    """

    single_artist_df = pd.read_csv(
        os.path.join(CSV_FOLDER, "SpeechBrain", "single_artist_cluster_df.csv"),
        index_col="track_id",
    )

    single_artist_df
    return (single_artist_df,)


@app.cell
def _(AUDIO_FOLDER, STEM_FOLDER, single_artist_df):
    single_artist_df["vocal_audio_path"] = single_artist_df[
        "vocal_audio_path"
    ].str.replace(STEM_FOLDER, "", regex=False)
    single_artist_df["audio_path"] = single_artist_df["audio_path"].str.replace(
        AUDIO_FOLDER, "", regex=False
    )
    single_artist_df.rename(
        columns={"audio_path": "song_path", "vocal_audio_path": "vocal_path"}
    ).drop(columns=["is_voiced", "is_voiced_check", "folder"])
    return


@app.cell
def _(RECALC_EMB, getTrimmedAudio, single_artist_df):
    if RECALC_EMB:
        X_df = single_artist_df.vocal_audio_path.apply(getTrimmedAudio)
        X = X_df.values
    return X, X_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Speechbrain Embedding Model
    """)
    return


@app.cell
def _():
    import torchaudio
    from speechbrain.inference.encoders import MelSpectrogramEncoder

    encoder = MelSpectrogramEncoder.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb-mel-spec"
    )
    return (encoder,)


@app.cell
def _():
    # embeddings = np.stack(triplet_df.vocal_audio_path.apply(
    #    lambda path: encoder.encode_waveform(getTrimmedAudio(path)).cpu().numpy().squeeze()
    # ).values)
    # embeddings_df = pd.DataFrame(embeddings, index=triplet_df.index)
    # embeddings_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notes

    n = 437 (number of unique artists) (testing batch needed?!, maybe same artist is enough for validation)
    d = (s,) (trimmed audio can have different lengths, at least 5*samplerate though)
    x = getTrimmedAudio(audiopath)

    n³ = 83,453,453 (way too many triplets)
    n*log(n) = ~1153 (minimum amount of triplets to train)
    (no grouping is necessary)

    K = ?

    Number of batches wanted: 1
    """)
    return


@app.cell
def _(encoder):
    encoder
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pre-Calculate K Dropouts
    """)
    return


@app.cell
def _(torch):
    def enable_dropout(model, p=0.2):
        for module in model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d)):
                module.p = p
                module.train()
    return (enable_dropout,)


@app.cell
def _(CSV_FOLDER, K, os):
    embeddings_path = os.path.join(CSV_FOLDER, "SpeechBrain", f"embeddings{K}.pt")
    return (embeddings_path,)


@app.cell
def _(K, RECALC_EMB, X, embeddings_path, enable_dropout, encoder, mo, torch):
    if RECALC_EMB:
        encoder.eval()
        enable_dropout(encoder)

        emb_dim = 192
        n_samples = len(X)
        all_embeddings = torch.zeros(n_samples, K, emb_dim)

        for mask_idx in mo.status.progress_bar(
            range(K),
            title="Getting dropout mask embeddings",
            show_eta=True,
            show_rate=True,
        ):
            for sample_idx, audio_tensor in enumerate(X):
                torch.manual_seed(mask_idx)
                audio_tensor = audio_tensor.to(encoder.device)
                emb = encoder.encode_waveform(audio_tensor)
                all_embeddings[sample_idx, mask_idx] = emb.detach().cpu()
        torch.save(all_embeddings, embeddings_path)
    else:
        all_embeddings = torch.load(embeddings_path)

    n = all_embeddings.shape[0]
    B = K  # has to be <= K fot proper computing
    d = all_embeddings.shape[2]
    return B, all_embeddings


@app.cell
def _(all_embeddings):
    all_embeddings.shape
    return


@app.cell
def _(batches):
    batches
    return


@app.cell
def _(all_embeddings, torch):
    torch.sum(all_embeddings[0, 0] - all_embeddings[0, 7])
    return


@app.cell
def _():
    """
    test_mask = 3
    text_x = 4
    torch.manual_seed(test_mask)
    test_emb = encoder.encode_waveform(X[text_x].to(encoder.device)).detach().cpu()
    torch.sum(all_embeddings[text_x, test_mask] - test_emb)
    """
    return


@app.cell
def _(RECALC_EMB, X, X_df, gc):
    if RECALC_EMB:
        del X
        del X_df
        gc.collect()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Cluster Embeddings

    Running the triplet selection over all triplet simultaniously is still very expensive:
    the number of vectors would be n³ (437³ = ~83mil)
    Which my 32GiB RAM did not like very much (sorry)

    This is why we need to break down the artists into existing groups, already present in the dataset
    The more Groups we have the less RAM we need. But too many groups may lead to a too biased model, so we need a balanced number.
    """)
    return


@app.cell
def _(G, all_embeddings, cut_tree_balanced, linkage, np):
    mean_embeddings = all_embeddings.mean(dim=1)
    Z = linkage(mean_embeddings, "ward")

    max_cluster_size = len(mean_embeddings) // G
    [balanced_cut_cluster_id, balanced_cut_cluster_level] = cut_tree_balanced(
        Z, max_cluster_size
    )

    # Get the number of clusters from the cut
    N_CLUSTERS = len(np.unique(balanced_cut_cluster_id))

    # cluster_embeddings = [
    #    all_embeddings[balanced_cut_cluster_id == g]
    #    for g in range(n_clusters_actual)
    # ]
    return N_CLUSTERS, balanced_cut_cluster_id, mean_embeddings


@app.cell
def _(gc, mean_embeddings):
    del mean_embeddings
    gc.collect()
    return


@app.cell
def _(N_CLUSTERS):
    N_CLUSTERS
    return


@app.cell
def _(balanced_cut_cluster_id, single_artist_df):
    single_artist_df["cluster"] = balanced_cut_cluster_id
    single_artist_df
    return


@app.cell
def _(single_artist_df):
    single_artist_df[single_artist_df.track_id == 99]
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate sets of tuples
    """)
    return


@app.cell
def _(balanced_cut_cluster_id, np, px):
    counts = np.bincount(balanced_cut_cluster_id, minlength=16)
    fig = px.bar(
        x=list(range(16)),
        y=counts,
        labels={"x": "Cluster ID", "y": "Count"},
        title="Distribution of Balanced Cut Cluster IDs",
    )

    fig.show()
    return


@app.cell
def _(N_CLUSTERS, balanced_cut_cluster_id, np, permutations, single_artist_df):
    all_triplet_idx = [
        np.array(
            list(
                permutations(
                    single_artist_df[balanced_cut_cluster_id == g].track_id.values,
                    3,
                )
            )
        )
        for g in range(N_CLUSTERS)
    ]
    return (all_triplet_idx,)


@app.cell
def _(all_triplet_idx):
    all_triplet_idx[0]
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Calculate uncertainty per tuple set
    """)
    return


@app.cell
def _(
    B,
    N_CLUSTERS,
    all_embeddings,
    all_triplet_idx,
    batches_path,
    mo,
    np,
    torch,
):
    # define functions
    def _d(x1, x2):
        """
        Calculate the distance between two embedding points
        Takes in either single points, each of shape (emb_dim) returns scalar value
        or the all masks (K, emb_dim) returns vector of length (K)
        or the whole batch (B, K, emb_dim) returns a matrix of shape (B, K)
        """
        return torch.linalg.vector_norm(x1 - x2, dim=-1)


    def _xi(xi, xj, xk):
        """
        Calculate the distance margin for a triplet
        Takes in either single points, each of shape (emb_dim) returns scalar value
        or the all masks (K, emb_dim) returns vector of length (K)
        or the whole batch (B, K, emb_dim) returns a matrix of shape (B, K)
        """
        d1 = torch.pow(_d(xi, xk), 2)
        d2 = torch.pow(_d(xi, xj), 2)
        return d1 - d2


    def _get_ut(t):
        """
        Calculate the distance margin for a triplet
        Takes in either single points, each of shape (emb_dim) returns scalar value
        or the all masks (K, emb_dim) returns vector of length (K)
        """
        ut = _xi(*t)
        return ut - ut.mean()


    def _get_Q(Bkm1, all_ut):
        """
        Get the orthogonal basis matrix of the matrix U from the triplet indicies
        """
        if len(Bkm1) == 0:
            return None

        # calculate U, by centering the matrix (no builtin method yet!)
        # https://stackoverflow.com/questions/76238864/center-a-tensor-in-pytorch
        # U*U^T = cov(U) => the covariance matrix (torch.cov(xiBkm1))
        # U will have to get the shape (k, k-1) to have k-1 columns and k rows
        # which is U is being transposed
        U = all_ut[Bkm1]
        U = U.T

        # calculate the orthonormal basis for span{us∣s∈Bk−1}
        return torch.linalg.qr(U, mode="reduced").Q


    def _get_suk2(Q, t, all_ut):
        """
        Essentially projects the new triplet vector ut onto the current orthonormal basis
        Calculates the orthogonal vector to this basis
        Returns the amount the deteminant would increase, if ut was added:
        det({U u ut}) = det(U) + ||~u||^2
        """
        # get ut as the zero mean vector to be appended to the matrix
        ut = all_ut[t]

        if Q == None:
            return (ut**2).sum()

        # calculate the normal of the span ui onto orthonomal basis Q
        proj = Q @ (Q.T @ ut)
        residual = ut - proj
        return (torch.pow(residual, 2)).sum()


    def _find_new_tk(Bkm1, all_ut):
        """
        Finds the best new triplet to add the the current Batch

        :param k: the size of the new batch (the old batch is of size k-1!)
        :param Bkm1: a list of triplet indicies from the list of all triplet values
        """
        Q = _get_Q(Bkm1, all_ut)

        most_fitting_t = 0
        max_suk2 = -1
        for t in range(len(all_ut)):
            # skip already choosen triplets
            if t in Bkm1:
                continue

            suk2 = _get_suk2(Q, t, all_ut)

            # check if the resulting ||~u||^2 is higher than the previous max
            if suk2 > max_suk2:
                max_suk2 = suk2
                best_t = t

        # return the best triplet index
        return best_t


    def get_best_batch(all_ut):
        batch = []
        for _ in mo.status.progress_bar(
            range(B),
            title="Getting best triplets for batch",
            show_eta=True,
            show_rate=True,
        ):
            batch.append(_find_new_tk(batch, all_ut))
        return batch


    best_batches = [
        all_triplet_idx[c][
            get_best_batch(
                torch.stack(
                    [
                        _get_ut([all_embeddings[j] for j in i])
                        for i in mo.status.progress_bar(
                            all_triplet_idx[c],
                            title="Calculating all ut",
                            show_eta=True,
                            show_rate=True,
                        )
                    ]
                )
            )
        ]
        for c in range(N_CLUSTERS)
    ]
    with open(batches_path, "wb") as f:
        np.save(f, np.array(best_batches).astype(np.float32))
    return


@app.cell
def _(batches_path, np):
    with open(batches_path, "rb") as fr:
        batches = np.load(fr)
    batches.shape
    return (batches,)


@app.cell
def _(batches, batches_path, np):
    with open(batches_path, "wb") as f:
        np.save(f, batches.astype(np.float32))
    return


if __name__ == "__main__":
    app.run()
