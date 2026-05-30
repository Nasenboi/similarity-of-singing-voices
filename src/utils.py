from typing import Union

import librosa
import numpy as np
import torch
from essentia import array
from essentia.standard import FFT, CartesianToPolar, FrameGenerator, MonoLoader, OnsetDetection, Onsets, Windowing


def get_trimmed_audio(
    audiopath: str, sr: int = 24_000, to_tensor=True, concat=True, min_duration=0
) -> Union[torch.Tensor, np.ndarray]:
    """Returns an audio sample array, with the silent parts cut out

    Args:
        audiopath (str): The path to the audio file
        sr (int): Sample Rate. Default is 24kHz
        to_tensor (bool): Either return tensor or np.array. Default is True
        concat (bool): Concatinate Speech into one array. Default is true
        min_duration (float): the minimum duration of each snippet in seconds. Default is 0 (all snippets should be used)

    Returns:
        torch.tensor: The audio sample array
    """

    y, sr = librosa.load(audiopath, sr=sr, mono=True)
    intervals = librosa.effects.split(y)
    trimmed_parts = []

    min_samples = min_duration * sr

    for start, end in intervals:
        if (end - start) >= min_samples:
            trimmed_parts.append(y[..., start:end])

    if concat:
        trimmed_parts = np.concatenate(trimmed_parts, axis=-1)

    if to_tensor:
        return torch.from_numpy(trimmed_parts)
    return trimmed_parts


def get_onsets_es(
    audio_file: str,
    method: str = "hfc",
    sample_rate: int = 44100,
    alpha: float = 0.1,
    silenceThreshold: float = 0.02,
    delay: int = 5,
    onset_pause: float = 1.0,
    round_down: bool = False,
) -> np.ndarray:
    """Detect onset timestamps in an audio file.
    Args:
        audio_file (str): path to audio file
        method (str): onset detection function ('complex', 'hfc', etc.). Default is hfc.
        sample_rate (int): audio sample rate. Default is 44100
        alpha (float): proportion of mean for adaptive threshold (higher = fewer onsets). Default is 0.1.
        silence_threshold (float): threshold for silence (higher = ignore more low-energy frames). Default is 0.02.
        delay (int): frames used for short-onset filter. Default is 5.
        onset_pause (float): minimum time interval in seconds between two onsets. Default is 1.
        round_down (bool): if true, rounds the onsets down to the nearest full second value.
    Returns:
        np.ndarray: numpy array of onset times in seconds (empty if none)
    """
    # Load audio
    audio = MonoLoader(filename=audio_file, sampleRate=sample_rate)()

    # Onset detection function
    od = OnsetDetection(method=method)
    w = Windowing(type="hann")
    fft = FFT()
    cart2pol = CartesianToPolar()

    features = []
    hop_size = 512
    for frame in FrameGenerator(audio, frameSize=1024, hopSize=hop_size):
        mag, phase = cart2pol(fft(w(frame)))
        features.append(od(mag, phase))

    # Compute onsets with adjustable threshold parameters
    onsets_algo = Onsets(
        frameRate=sample_rate / float(hop_size),
        alpha=alpha,
        silenceThreshold=silenceThreshold,
        delay=delay,
    )
    onsets = onsets_algo(array([features]), [1])

    # --- Round down to nearest full second ---
    if round_down and len(onsets) > 0:
        onsets = np.floor(onsets)

    # --- Enforce minimum pause between onsets ---
    if onset_pause is not None and len(onsets) > 0:
        filtered_onsets = [onsets[0]]
        for t in onsets[1:]:
            if t - filtered_onsets[-1] >= onset_pause:
                filtered_onsets.append(t)
        onsets = np.array(filtered_onsets)

    return onsets
