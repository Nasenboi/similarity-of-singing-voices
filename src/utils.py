import librosa
import numpy as np
import torch


def get_trimmed_audio(audiopath: str, sr: int = 24_000) -> torch.Tensor:
    """Returns an audio sample array, with the silent parts cut out

    Args:
        audiopath (str): The path to the audio file
        sr (int): Sample Rate. Default is 24kHz

    Returns:
        torch.tensor: The audio sample array
    """
    y, sr = librosa.load(audiopath, sr=sr, mono=True)
    intervals = librosa.effects.split(y)
    trimmed_parts = []
    for start, end in intervals:
        trimmed_parts.append(y[..., start:end])
    return torch.from_numpy(np.concatenate(trimmed_parts, axis=-1))