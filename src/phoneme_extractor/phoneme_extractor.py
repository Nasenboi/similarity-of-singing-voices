import os
from typing import List, Optional, Tuple

import langcodes
import marimo
import numpy as np
import pandas as pd
import torch
from bournemouth_aligner import PhonemeTimestampAligner
from pydantic import BaseModel
from qwen_asr import Qwen3ASRModel

from ..utils import get_trimmed_audio


class PhonemeDataRow(BaseModel):
    phoneme: str
    file_id: Optional[str]
    snippet_id: Optional[int]
    start_ms: float
    end_ms: float
    duration_ms: float
    confidence: float


def load_data(load_path: str, pickled: bool = False) -> Tuple[pd.DataFrame, np.array]:
    """Load the the phoneme and row data

    Args:
        load_path (str): Folder of the phoneme data
        pickled (bool, optional): Numpy pickle. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, np.array]: The phoneme data
    """
    rows = load_rows(os.path.join(load_path, "phoneme_rows.parquet"))
    phonemes = load_phonemes(os.path.join(load_path, "phonemes.npy"), pickled)

    return rows, phonemes


def load_rows(path: str) -> pd.DataFrame:
    """Loads the phoneme DataFrame

    Args:
        path (str): Path to the DataFrame

    Returns:
        pd.DataFrame: Phoneme DataFrame
    """
    return pd.read_parquet(path, index="phoneme_id")


def load_phonemes(path: str, pickled: bool = False) -> np.array:
    """Loads the phoneme numpy arrays

    Args:
        path (str): Path to the Numpy array
        pickled (bool, optional): Numpy pickle. Defaults to False.

    Returns:
        np.array: The Phoneme numpy array
    """
    return np.load(path, allow_pickle=pickled)


class PhonemeExtractor:
    """
    The Phoneme Extractor Class
    Contains utilites to extract phonemes from singing audio files.

    Contains marimo progress bars, so you should use it in marimo notebooks.

    This class uses a diverse set of pretrained models:
    """

    def __init__(
        self,
        asr_model_path: str,
        fa_model_path: str,
        sample_rate: int = 16_000,
        device: torch.device = None,
        min_sippet_duration: float = 2.0,
        min_phoneme_duration_ms: float = 40,
        min_phoneme_confidence: float = 0.0,
    ):
        """PhonemeExtractor Constructor

        Baseline for the min phoneme duration:
        Elias, F. (2017). An Acoustic Analysis of Vowel Duration in Wolaytta Doonaa. M.A. Thesis.

        Args:
            asr_model_path (str): Path to the Qwen3ASRModel checkpoint
            fa_model_path (str): Path to the PhonemeTimestampAligner checkpoint
            sample_rate (int, optional): Audio Sample Rate. Defaults to 16_000.
            device (torch.device, optional): Torch Device. Defaults to "cuda" if it exists.
            min_sippet_duration (float, optional): Minimum duration in seconds for audio snippets. Defaults to 2.0.
            min_phoneme_duration_ms (float, optional): Minimum duration in milliseconds for phonemes. Defaults to 40. !! Not implemented yet !!
            min_phoneme_confidence (float, optional): Minimum model confidence level for phoneme. Defaults to 0.0 (all). !! Not implemented yet !!
        """
        # Universal values
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.min_snippet_duration = min_sippet_duration
        self.min_phoneme_duration = min_phoneme_duration_ms
        self.min_phoneme_confidence = min_phoneme_confidence

        # Automatic Speech Recognition
        self.asr_model = Qwen3ASRModel.from_pretrained(
            asr_model_path,
            dtype=torch.bfloat16,
            device_map=str(self.decive),
            max_inference_batch_size=32,
            max_new_tokens=256,
        )

        # Forced Alignment
        self.forced_aligner = PhonemeTimestampAligner(
            cupe_ckpt_path=fa_model_path, device=self.device, bad_confidence_threshold=1
        )

    def process_batch(
        self, files, ids: List[str] = None, save_path: str = None, allow_pickle: bool = False
    ) -> Tuple[List[PhonemeDataRow], List[np.array]]:
        """Processes a batch of audio files

        Args:
            files (_type_): A list of audio file paths
            ids (List[str], optional): An optional list of file ids, should be the same length as files. Defaults to None.
            save_path (str, optional): An optional path to save the data to. Defaults to None.
            allow_pickle (bool, optional): Allow Numpy pickle algorithm. Defaults to False.

        Returns:
            Tuple[List[PhonemeDataRow], List[np.array]]: The phoneme data
        """
        rows = []
        phonemes = []
        for idx, path in marimo.status.progress_bar(enumerate(files), title="Extracting Phonemes for all Files..."):
            file_id = ids[idx] if ids is not None and len(ids) > idx else None
            r, p = self.process_single_file(path, file_id)
            rows += r
            phonemes += p

        if save_path is not None:
            self.__save_rows(rows, os.path.join(save_path, "phoneme_rows.parquet"))
            self.__save_phonemes(phonemes, os.path.join(save_path, "phonemes.npy"))
        return rows, phonemes

    def process_single_file(self, file_path: str, file_id: str = None) -> Tuple[List[PhonemeDataRow], List[np.array]]:
        """Processes a single speech / singing audio file

        Args:
            file_path (str): The path to the audio file
            file_id (str, optional): An optional file id. Defaults to None.

        Returns:
            Tuple[List[PhonemeDataRow], List[np.array]]: The phoneme data
        """
        try:
            y_snippets = get_trimmed_audio(
                file_path, sr=self.sample_rate, to_tensor=False, concat=False, min_duration=self.min_snippet_duration
            )
            text_snippets, lang = self.__extract_text(y_snippets=y_snippets)
            rows, phonemes = self.__get_phonemes(y_snippets, text_snippets, file_id, lang)

            return rows, phonemes
        except Exception as e:
            print(f"Error processing f{file_id if  file_id is not None else file_path}:\n{e}")

    def __extract_text(self, y_snippets: List[np.array]) -> Tuple[List[str], str]:
        if len(y_snippets) == 0:
            return None
        asr_results = [self.asr_model.transcribe(audio=(s, self.sample_rate))[0] for s in y_snippets]
        asr_texts = [asr.text for asr in asr_results]
        asr_language = self.__to_language_code(asr_results[0].language)

        return asr_texts, asr_language

    def __get_phonemes(
        self, y_snippets: List[np.array], text_snippets: List[str], file_id: str, lang: str
    ) -> Tuple[List[PhonemeDataRow], List[np.array]]:
        self.__set_aligner_language(lang)
        audios = []
        for y in y_snippets:
            y_tensor = torch.tensor(y).to(self.device)
            y_stereo = y_tensor.unsqueeze(0).expand(2, -1)
            audios.append(self.forced_aligner.load_audio(y_stereo, sr=self.sample_rate))
        fa_results = self.forced_aligner.process_sentences_batch(text_snippets, audios)

        rows = []
        phonemes = []
        for idx, res in enumerate(fa_results):
            phoneme_ts = [ts for seg in res["segments"] for ts in seg["phoneme_ts"]]
            r, p = self.__process_aligner_result(y_snippets[idx], phoneme_ts, file_id, idx)
            rows += r
            phonemes += p

        return rows, phonemes

    def __set_aligner_language(self, lang: str = "en"):
        if lang != self.forced_aligner.lang:
            self.forced_aligner.phonemizer.set_backend(language=lang)
            self.forced_aligner.lang = lang

    def __process_aligner_result(
        self, y: np.array, phoneme_ts: List[dict], file_id: str, snippet_id: int
    ) -> Tuple[List[PhonemeDataRow], List[np.array]]:
        rows = []
        phonemes = []
        for phoneme in phoneme_ts:
            rows.append(
                PhonemeDataRow(
                    file_id=file_id,
                    snippet_id=snippet_id,
                    start_ms=phoneme["start_ms"],
                    end_ms=phoneme["end_ms"],
                    duration_ms=phoneme["end_ms"] - phoneme["start_ms"],
                    confidence=phoneme["confidence"],
                )
            )
            phonemes.append(self.__get_phoneme(y, phoneme["start_ms"], phoneme["end_ms"]))
        return rows, phonemes

    def __get_phoneme(self, y: np.array, start_ms: float, end_ms: float) -> np.array:
        start_idx = self.__ms_to_idx(start_ms, len(y))
        end_idx = self.__ms_to_idx(end_ms, len(y))
        return y[start_idx:end_idx]

    def __ms_to_idx(self, ms: float, arr_len: int) -> int:
        idx = int(ms * self.sample_rate)
        return max(min(idx, arr_len), 0)

    def __save_rows(self, rows: List[PhonemeDataRow], path: str):
        df = pd.DataFrame([r.__dict__ for r in rows])
        df.index.name = "phoneme_id"
        df.to_parquet(path)

    def __save_phonemes(self, phonemes: List[np.array], path: str, allow_pickle: bool = False):
        np.save(path, np.array(phonemes, dtype=object), allow_pickle=allow_pickle)

    @staticmethod
    def __to_language_code(lang: str) -> str:
        code = langcodes.find(lang)
        return code.language
