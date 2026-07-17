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

    return (
        CSV_FOLDER,
        MODEL_FOLDER,
        get_trimmed_audio,
        librosa,
        mo,
        np,
        os,
        pathlib,
        pd,
        plt,
        torch,
    )


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
    return DEVICE, SAMPLE_RATE, VOCAL_PATH


@app.cell
def _(VOCAL_PATH, pathlib):
    TXT_PATH = pathlib.Path(VOCAL_PATH).with_suffix(".txt")
    return


@app.cell
def _(SAMPLE_RATE, VOCAL_PATH, get_trimmed_audio, np):
    # y, sr = librosa.load(VOCAL_PATH, sr=SAMPLE_RATE, mono=True)
    y_snippets = get_trimmed_audio(VOCAL_PATH, sr=SAMPLE_RATE, to_tensor=False, concat=False, min_duration=2)
    y = np.concatenate(y_snippets, axis=-1)
    return y, y_snippets


@app.cell
def _(y_snippets):
    len(y_snippets)
    return


@app.cell
def _(SAMPLE_RATE, mo, y_snippets):
    mo.audio(y_snippets[-1], SAMPLE_RATE)
    return


@app.cell
def _(SAMPLE_RATE, librosa, np, plt, y):
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, fmax=8000)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=SAMPLE_RATE, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel-frequency spectrogram")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Automatic Speech Recognition

    - Python Package from [GitHub](https://github.com/QwenLM/Qwen3-ASR)
    """)
    return


@app.cell
def _():
    from qwen_asr import Qwen3ASRModel

    return (Qwen3ASRModel,)


@app.cell
def _(MODEL_FOLDER, os):
    asr_model_path = os.path.join(MODEL_FOLDER, "Qwen-ASR-1.7B")
    return (asr_model_path,)


@app.cell
def _(Qwen3ASRModel, asr_model_path, torch):
    asr_model = Qwen3ASRModel.from_pretrained(
        asr_model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    return (asr_model,)


@app.cell
def _():
    # unload
    """
    del asr_model
    # asr_model = None
    torch.cuda.empty_cache()
    """
    return


@app.cell
def _(SAMPLE_RATE, asr_model, y_snippets):
    """
    if not os.path.exists(TXT_PATH):
        asr_result = asr_model.transcribe(audio=(y, SAMPLE_RATE))[0]
        asr_text = asr_result.text
        with open(TXT_PATH, "w") as f:
            f.write(asr_result.text)
    else:
        print("Path exists")
        with open(TXT_PATH, "r") as f:
            asr_text = f.read()
    """

    asr_results = [asr_model.transcribe(audio=(s, SAMPLE_RATE))[0] for s in y_snippets]
    asr_texts = [asr.text for asr in asr_results]
    asr_texts[0]
    return asr_results, asr_texts


@app.cell
def _(SAMPLE_RATE, mo, y_snippets):
    mo.audio(y_snippets[0], rate=SAMPLE_RATE)
    return


@app.cell
def _(asr_result, asr_results):
    import langcodes

    def to_language_code(lang: str) -> str:
        code = langcodes.find(lang)
        return f"{code.language}"

    try:
        asr_lanugage = (
            to_language_code(asr_results[0].language)
            if asr_result.language is not None
            else to_language_code("english")
        )
    except Exception as e:
        asr_lanugage = to_language_code("english")
    asr_lanugage
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forced Speech Alignment for Phenome Snippets

    - Python Package from [GitHub](https://github.com/tabahi/bournemouth-forced-aligner)

    Rehman, A., Cai, J., Zhang, J.-J., & Yang, X. (2025). BFA: Real-time Multilingual Text-to-speech Forced Alignment. https://arxiv.org/abs/2509.23147
    """)
    return


@app.cell
def _():
    from bournemouth_aligner import PhonemeTimestampAligner

    return (PhonemeTimestampAligner,)


@app.cell
def _(MODEL_FOLDER, os):
    fa_model_path = os.path.join(
        MODEL_FOLDER,
        "bournemouth",
        "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt",
    )
    return (fa_model_path,)


@app.cell
def _(PhonemeTimestampAligner, fa_model_path):
    aligner = PhonemeTimestampAligner(preset="asr_lanugage", cupe_ckpt_path=fa_model_path)
    return (aligner,)


@app.cell
def _(DEVICE, torch, y_snippets):
    y_tensors = [torch.tensor(s).to(DEVICE) for s in y_snippets]
    wav_tensors = [ten.unsqueeze(0).expand(2, -1) for ten in y_tensors]
    return (wav_tensors,)


@app.cell
def _(SAMPLE_RATE, aligner, wav_tensors):
    audios = [aligner.load_audio(ten, sr=SAMPLE_RATE) for ten in wav_tensors]
    return (audios,)


@app.cell
def _(aligner, asr_texts, audios):
    fa_restult = aligner.process_sentences_batch(asr_texts, audios)
    return (fa_restult,)


@app.cell
def _(fa_restult):
    fa_restult
    return


@app.cell
def _(y_snippets):
    y_snippets[0].shape
    return


@app.cell
def _(fa_restult):
    fa_restult[0]["segments"][0]
    return


@app.cell
def _(fa_restult):
    fa_restult[0]["segments"][0]["phoneme_ts"]
    return


@app.cell
def _(plt, torch):
    def plot_mel_phonemes(mel, compress_framesed, save_path="mel_phonemes.png"):
        """
        Plot mel spectrogram with phoneme IDs overlaid directly on the spectrogram

        Args:
            mel: Mel spectrogram tensor [frames, mel_bins]
            compress_framesed: List of [phoneme_id, count] pairs representing phoneme alignment per frame
            save_path: Path to save the plot
        """
        assert mel.dim() == 2, f"Expected 2D mel tensor, got {mel.dim()}D"
        phn_frame_ids = [phoneme_id for phoneme_id, _ in compress_framesed]
        phn_frame_counts = [count for _, count in compress_framesed]

        # Create single plot - make it twice as wide
        fig, ax = plt.subplots(1, 1, figsize=(30, 8))

        # Convert mel to numpy for plotting
        mel_np = mel.cpu().numpy() if isinstance(mel, torch.Tensor) else mel

        # Add statistics to title instead of overlaying on spectrum
        unique_phonemes = len(set(phn_frame_ids))
        stats_text = f"Frames: {sum(phn_frame_counts)} | Unique Phonemes: {unique_phonemes} | Mel Bins: {mel.shape[1]}"
        title_text = f"Mel Spectrogram with Phoneme Alignment\n{stats_text}"

        # Plot mel spectrogram
        im = ax.imshow(
            mel_np.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_ylabel("Mel Bins")
        ax.set_xlabel("Frame Index")
        ax.set_title(title_text)
        plt.colorbar(im, ax=ax, label="Magnitude")

        # Overlay phoneme information
        frame_pos = 0
        for phn_id, count in zip(phn_frame_ids, phn_frame_counts):
            # Draw vertical boundary lines (except for first segment)
            if frame_pos > 0:
                ax.axvline(
                    x=frame_pos - 0.5,
                    color="red",
                    linestyle="-",
                    alpha=0.8,
                    linewidth=2,
                )

            # Add phoneme ID text at the top of the spectrogram
            if count > 1:  # Only add text if segment is wide enough
                text_x = frame_pos + count / 2
                text_y = mel.shape[1] - 2  # Near the top of the mel bins

                # Add text with background for visibility
                ax.text(
                    text_x,
                    text_y,
                    str(phn_id),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
                )

            frame_pos += count

        plt.tight_layout()
        plt.show()
        return save_path

    return (plot_mel_phonemes,)


@app.cell
def _(aligner, audios, fa_restult, plot_mel_phonemes):
    mel_spec = aligner.extract_mel_spectrum(
        audios[0].cpu()[0].unsqueeze(0),
        wav_sample_rate=aligner.resampler_sample_rate,
    )

    # --- Phoneme → frame mapping ---
    seg = fa_restult[0]["segments"][0]
    segment_duration = seg["end"] - seg["start"]  # in seconds
    total_frames = mel_spec.shape[0]
    frames_per_second = total_frames / segment_duration

    frames_assorted = aligner.framewise_assortment(
        aligned_ts=seg["phoneme_ts"],
        total_frames=total_frames,
        frames_per_second=frames_per_second,
        gap_contraction=0,
        select_key="phoneme_id",
    )

    frames_assorted = [aligner.phoneme_id_to_label.get(pid, "...") for pid in frames_assorted]

    compress_framesed = aligner.compress_frames(frames_assorted)

    plot_mel_phonemes(mel_spec, compress_framesed)
    return


@app.cell
def _(SAMPLE_RATE, mo, y_snippets):
    mo.audio(y_snippets[0], rate=SAMPLE_RATE)
    return


@app.cell
def _(SAMPLE_RATE):
    SAMPLE_RATE
    return


if __name__ == "__main__":
    app.run()
