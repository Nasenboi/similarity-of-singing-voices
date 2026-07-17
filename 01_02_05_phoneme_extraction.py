import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import Python Packages
    """)
    return


@app.cell
def _():
    import os
    import pathlib

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    from src.globals import (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        MODEL_FOLDER,
        STEMS_FOLDER,
        TRACKS_PATH,
        UVR_MODEL_PATH,
    )
    from src.utils import get_trimmed_audio

    return CSV_FOLDER, DATASET_FOLDER, MODEL_FOLDER, mo, os, pd, torch


@app.cell(hide_code=True)
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
def _(torch, track_df):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAMPLE_RATE = 16_000
    SAMPLE_TRACK = track_df.sample(n=1).iloc[0]
    SONG_PATH = SAMPLE_TRACK.song_path
    VOCAL_PATH = SAMPLE_TRACK.vocal_path
    VOCAL_PATH
    return DEVICE, SAMPLE_RATE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phoneme Extraction
    """)
    return


@app.cell
def _():
    from src.phoneme_extractor.phoneme_extractor import PhonemeExtractor, load_data

    return (PhonemeExtractor,)


@app.cell
def _(DATASET_FOLDER, MODEL_FOLDER, os):
    phoneme_save_path = os.path.join(DATASET_FOLDER, "fma_large_phonemes")
    asr_model_path = os.path.join(MODEL_FOLDER, "Qwen-ASR-1.7B")
    fa_model_path = os.path.join(
        MODEL_FOLDER,
        "bournemouth",
        "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt",
    )
    return asr_model_path, fa_model_path, phoneme_save_path


@app.cell
def _(DEVICE, PhonemeExtractor, SAMPLE_RATE, asr_model_path, fa_model_path):
    phoneme_extractor = PhonemeExtractor(
        asr_model_path=asr_model_path,
        fa_model_path=fa_model_path,
        device=DEVICE,
        sample_rate=SAMPLE_RATE,
    )
    return (phoneme_extractor,)


@app.cell
def _():
    # rows, phonemes = phoneme_extractor.process_single_file(file_id=str(SAMPLE_TRACK.name), file_path=VOCAL_PATH)
    # rows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Whole Batch Processing
    """)
    return


@app.cell
def _(phoneme_extractor, phoneme_save_path, track_df):
    rows, phonemes = phoneme_extractor.process_batch(
        files=list(track_df.vocal_path.values),
        ids=list(track_df.index.values),
        save_path=phoneme_save_path,
        allow_pickle=True,
    )
    return (rows,)


@app.cell
def _(pd, rows):
    df = pd.DataFrame([r.__dict__ for r in rows])
    df.index.name = "phoneme_id"
    df
    return


if __name__ == "__main__":
    app.run()
