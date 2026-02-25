import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/music-cluster-3D.slides.json",
)


@app.cell
def _():
    from os import getenv, path
    from random import choice

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import umap
    from dotenv import load_dotenv
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    from transformers import Data2VecAudioModel, Wav2Vec2Processor

    import utils

    load_dotenv()
    return (
        Data2VecAudioModel,
        StandardScaler,
        Wav2Vec2Processor,
        choice,
        getenv,
        librosa,
        mo,
        np,
        path,
        pd,
        torch,
        tqdm,
        umap,
        utils,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create a Cluster of Musical Tracks

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
    return AUDIO_FOLDER, TRACKS_PATH


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


    Possible solutions:

    [Open L3](https://github.com/marl/openl3)

    [Deep Siamese Network](https://arxiv.org/abs/2006.00572)
    """)
    return


@app.cell
def _(tracks_df):
    train_df = tracks_df[tracks_df["set", "split"] == "training"].copy()
    return (train_df,)


@app.cell
def _(Data2VecAudioModel, Wav2Vec2Processor, librosa, torch, tqdm):
    class MusicEmbeddingExtractor:
        def __init__(self, device=None):
            self.device = (
                device
                if device
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.processor = Wav2Vec2Processor.from_pretrained(
                "facebook/data2vec-audio-base-960h"
            )
            self.model = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1")
            self.model.to(self.device)
            self.model.eval()

        def process_audio_file(self, audio_path, sr=16000):
            """Process a single audio file"""
            audio, sr = librosa.load(audio_path, sr=sr)

            # Process through the model
            inputs = self.processor(
                audio, sampling_rate=sr, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # Stack all hidden states
            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

            # Option 1: Mean pooling across time
            time_reduced = all_layer_hidden_states.mean(dim=1)  # [13, 768]

            # Option 2: Weighted average across layers (learnable or simple mean)
            # Simple mean across layers:
            embedding = time_reduced.mean(dim=0)  # [768]

            return embedding.cpu().numpy()

        def process_audio_array(self, audio_array, sr=16000):
            """Process audio directly from numpy array"""
            inputs = self.processor(
                audio_array, sampling_rate=sr, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

            # You can experiment with different pooling strategies:
            # 1. Mean pooling (simplest)
            embedding = all_layer_hidden_states.mean(dim=1).mean(dim=0)

            # 2. CLS token if available (not in this model)
            # 3. Max pooling
            # embedding = all_layer_hidden_states.max(dim=1)[0].mean(dim=0)

            return embedding.cpu().numpy()

        def batch_process(self, audio_paths, sr=16000, batch_size=1):
            """Process multiple audio files"""
            embeddings = []
            for i in tqdm(range(0, len(audio_paths), batch_size)):
                batch_paths = audio_paths[i : i + batch_size]
                batch_embeddings = []

                for path in batch_paths:
                    emb = None
                    try:
                        emb = self.process_audio_file(path, sr)
                    except:
                        pass
                    batch_embeddings.append(emb)

                embeddings.extend(batch_embeddings)

            return embeddings
    return (MusicEmbeddingExtractor,)


@app.cell
def _(MusicEmbeddingExtractor):
    extractor = MusicEmbeddingExtractor()
    return (extractor,)


@app.cell
def _(extractor, train_df):
    music_embeddings = extractor.batch_process(train_df["audio_path"])
    return (music_embeddings,)


@app.cell
def _(StandardScaler, music_embeddings):
    scaler = StandardScaler()
    mask = [e is not None for e in music_embeddings]
    music_embeddings_scaled = scaler.fit_transform(
        [e for e in music_embeddings if e is not None]
    )
    return mask, music_embeddings_scaled


@app.cell
def _(music_embeddings_scaled, umap):
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=5,
        min_dist=0.1,
        metric="cosine",
    )

    umap_embeddings = reducer.fit_transform(music_embeddings_scaled)
    return (umap_embeddings,)


@app.cell
def _(mask, np, train_df, umap_embeddings):
    train_df["UMAP_1"] = np.nan
    train_df["UMAP_2"] = np.nan
    train_df["UMAP_3"] = np.nan
    train_df.loc[mask, "UMAP_1"] = umap_embeddings[:, 0]
    train_df.loc[mask, "UMAP_2"] = umap_embeddings[:, 1]
    train_df.loc[mask, "UMAP_3"] = umap_embeddings[:, 2]
    return


@app.cell
def _(train_df):
    train_df
    return


@app.cell
def _(train_df):
    train_df.to_csv("tracks_df_embedding_3D.csv")
    return


@app.cell
def _(pd, utils):
    df = utils.load("tracks_df_embedding_3D.csv")
    new_columns = list(df.columns)
    new_columns[-4:] = [
        ("track", "audio_path"),
        ("UMAP", "X"),
        ("UMAP", "Y"),
        ("UMAP", "Z"),
    ]
    df.columns = pd.MultiIndex.from_tuples(new_columns)
    return (df,)


@app.cell
def _(df):
    df.columns()
    return


if __name__ == "__main__":
    app.run()
