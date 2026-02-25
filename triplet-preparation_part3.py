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
    import librosa as lr
    import numpy as np

    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    load_dotenv(os.path.join(BASEDIR, ".env"))
    return lr, np, os, pd


@app.cell
def _(os):
    RANDOM_STATE = 42
    DATASET_FOLDER = os.getenv("DATASET_FOLDER")
    CSV_FOLDER = os.getenv("CSV_FOLDER")
    AUDIO_FOLDER = os.path.join(DATASET_FOLDER, "fma_large")
    STEM_FOLDER = os.path.join(DATASET_FOLDER, "fma_large_stems")
    VOCALS_FILE_NAME = "vocals.mp3"
    return CSV_FOLDER, STEM_FOLDER, VOCALS_FILE_NAME


@app.cell
def _(CSV_FOLDER, os, pd):
    triplet_df = pd.read_csv(
        os.path.join(CSV_FOLDER, "triplet_df_voiced.csv"), index_col="track_id"
    )
    triplet_df
    return (triplet_df,)


@app.cell
def _(STEM_FOLDER, VOCALS_FILE_NAME, os, triplet_df):
    def getVocalPath(track_id: int):
        track_id_zp = "{0:06d}".format(track_id)
        vocal_path = os.path.join(
            STEM_FOLDER, track_id_zp[:3], track_id_zp, VOCALS_FILE_NAME
        )
        return vocal_path


    triplet_df["vocal_audio_path"] = triplet_df.index.to_series().apply(
        getVocalPath
    )
    return


@app.cell
def _(lr, np):
    SAMPLE_RATE = 16_000
    AUDIO_LENGTH_S = 30
    MFCC_SETTINGS = {
        "n_mels": 128,
        "n_mfcc": 20,
        "sr": SAMPLE_RATE,
        "fmax": SAMPLE_RATE / 2,
        "fmin": 80,
        "hop_length": 512,
    }
    ENERGY_THESHOLD = 0.025
    MIN_VOICE_THRESHOLD = 0.2


    def getMFCCFeatures(audiopath: str):
        try:
            y, sr = lr.load(audiopath)
            hop_length = MFCC_SETTINGS["hop_length"]

            # 1. Compute RMS energy per frame (same hop_length as MFCCs)
            rms = lr.feature.rms(y=y, hop_length=hop_length)[0]  # shape: (T,)

            # 2. Simple energy threshold: 1% of the maximum RMS
            thresh = ENERGY_THESHOLD * np.max(rms)
            voiced_frames = rms > thresh

            # 3. Check if voiced proportion is sufficient (≥10% of frames)
            total_frames = len(rms)
            voiced_ratio = np.sum(voiced_frames) / total_frames
            if voiced_ratio < MIN_VOICE_THRESHOLD:
                #print(voiced_ratio, rms)
                return None  # too much silence

            # 4. Compute MFCCs
            mfccs = lr.feature.mfcc(y=y, **MFCC_SETTINGS)

            feature_vector = []
            for coeff_sequence in mfccs:
                voiced_vals = coeff_sequence[voiced_frames]
                feature_vector.extend([np.mean(voiced_vals), np.std(voiced_vals)])

            return feature_vector
        except Exception as e:
            print(e)
            return None


    n_mfcc = MFCC_SETTINGS["n_mfcc"]
    stats = ["mean", "std"]
    FEATURE_NAMES = [
        f"feature_MFCC.{b + 1}_{s}" for b in range(n_mfcc) for s in stats
    ]
    return FEATURE_NAMES, getMFCCFeatures


@app.cell
def _(FEATURE_NAMES, getMFCCFeatures, pd, triplet_df):
    feature_arrays = triplet_df.vocal_audio_path.apply(getMFCCFeatures)

    valid_mask = feature_arrays.notna()
    feature_arrays = feature_arrays[valid_mask]

    features_df = pd.DataFrame(
        feature_arrays.tolist(), index=feature_arrays.index, columns=FEATURE_NAMES
    )

    features_df
    return


if __name__ == "__main__":
    app.run()
