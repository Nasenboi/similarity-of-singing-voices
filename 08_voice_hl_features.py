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
    import os

    import marimo as mo
    import numpy as np
    import pandas as pd

    from src.FMA.utils import get_audio_path, load
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

    return CSV_FOLDER, mo, os, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Load the Dataset
    Song Features already present
    """)
    return


@app.cell
def _(CSV_FOLDER, os, pd):
    track_df = pd.read_csv(
        os.path.join(
            CSV_FOLDER,
            "LargeDataset",
            "additional_features",
            "high_level_song_features.csv",
        ),
        index_col="track_id",
    )
    track_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Get Additional Features

    Most of them will be estimated by machine learning techniques

    ## High Level Singer Information
    - age
    - gender
    - accent
    - language
    - timbre
    """)
    return


@app.cell
def _(os):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import json

    import librosa as lr
    import tensorflow as tf

    tf.debugging.set_log_device_placement(True)
    print(tf.config.list_physical_devices("GPU"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Estimated Speaker Gender
    - Gender Estimation Model from [GitHub](https://github.com/JaesungHuh/voice-gender-classifier) thanks to "JaesungHuh"
    """)
    return


@app.cell
def _():
    # https://github.com/TaoRuijie/ECAPA-TDNN
    # https://github.com/JaesungHuh/voice-gender-classifier
    """
    import torch
    import sys
    sys.path.append("src/submodules/voice-gender-classifier/")
    from model import ECAPA_gender

    genderModelPath = os.path.join(MODEL_FOLDER, "gender")
    model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier", cache_dir=genderModelPath)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def getVoiceGender(path: str):
        realPath = path.replace(os.getenv("DRIVE"), "/data")
        # audio = model.load_audio(realPath).to(device)
        audio = get_trimmed_audio(audiopath=realPath, sr=16000).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.forward(audio)
            probs = torch.softmax(logits, dim=1)[0]

        return pd.Series({
            "pred_gender": model.pred2gender[probs.argmax().item()],
            "pred_p_male": probs[0].item(),
            "pred_p_female": probs[1].item(),
        })

    track_df[["pred_gender", "pred_p_male", "pred_p_female"]] = track_df.vocal_path.apply(getVoiceGender)
    track_df
    """
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Estimated Speaker Age
    - AgeRegressionPipeline from [GitHub](https://github.com/griko/voice-age-regression) thanks to "griko"
    - Age regression model from [HiggingFace](https://huggingface.co/griko/age_reg_svr_ecapa_librosa_voxceleb2) thanks to  "griko"

    Koushnir, G., Fire, M., Alpert, G. F., & Kagan, D. (2025). VANPY: Voice Analysis Framework. https://arxiv.org/abs/2502.17579
    """)
    return


@app.cell
def _():
    # https://github.com/TaoRuijie/ECAPA-TDNN
    # https://github.com/griko/voice-age-regression
    # https://huggingface.co/griko/age_reg_svr_ecapa_librosa_voxceleb2
    """
    import torch
    import sys
    import pipeline
    sys.path.append("src/submodules/voice-age-regression/src/")
    from voice_age_regressor import AgeRegressionPipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ageModelPath = os.path.join(MODEL_FOLDER, "age")
    model = AgeRegressionPipeline.from_pretrained(
        ageModelPath, device=device
    )

    def getVoiceAge(path: str):
        realPath = path.replace(os.getenv("DRIVE"), "/data")

        audio = get_trimmed_audio(audiopath=realPath, sr=16000).unsqueeze(0).to(device)
        with torch.no_grad():
            inputs = {
                "inputs": audio,
                "wav_lens": torch.tensor([1.0], dtype=torch.float32).to(device)
            }
            features = model.forward(inputs)
            pred_age = model.postprocess(features)[0]
            pred_age_no_trim = model(realPath)[0]

        return pd.Series({
            "pred_age": pred_age,
            "pred_age_no_trim": pred_age_no_trim
        })

    track_df[["pred_age", "pred_age_no_trim"]] = track_df.vocal_path.apply(getVoiceAge)
    track_df
    """
    return


if __name__ == "__main__":
    app.run()
