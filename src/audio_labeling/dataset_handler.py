import os
from datetime import datetime
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import sounddevice as sd

from ..globals import CSV_FOLDER


class DatasetHandler:
    """
    Class to handle the dataset operations for the audio labeling process
    """

    def __init__(self, start_pos: int = -1, dataset_path: str = None):
        """
        Init the class
        Load the dataset and the first row to edit

        Args:
            start_pos (int): The position of the starting index. Defaults to -1 (the first unchecked row)
            dataset_path (str): A custom string to the dataset path. Defaults to None => the latest dataset
        """
        self._subpath = "LargeDataset"
        self._dataset = pd.DataFrame()
        self.current_index = start_pos
        self.current_row = {}

        self._load_current_dataset(dataset_path)

        if start_pos == -1:
            self.current_index = 0
            self._step_forward()
        else:
            self._get_row_infos()

    # -- Public Methods --

    # Dataset Methods
    def save(self):
        """generates a new path and saves the current state of the dataset
        """
        self._dataset.to_csv(self._generate_new_dataset_path())

    def get_progress(self) -> dict:
        """Get the labeling progres
        Returns:
            dict: progress informations
        """
        total = len(self._dataset)
        checked = len(self._dataset[self._dataset.checked])
        percent = (checked / total * 100) if total > 0 else 0
        return {
            "total": total,
            "checked": checked,
            "percent": percent
        }

    # Row Methods
    def set_row(self, step_forward: bool = True) -> bool:
        """Sets the row with the current row information.

        Args:
            step_forward (bool): If True, the current index will be updated. Defaults to True.

        Returns:
            bool: Returns False if the current index did not update.
        """
        self._insert_row_infox()
        if step_forward:
            return self._step_forward()
        return True
    
    def navigate(self, step: int = 1):
        """navigates the dataset using a specified step size
        
        Args:
            step (int): The step size. Defaults to 1.
        """
        self.current_index = self.current_index + step
        self.current_index = max(0, min(self.current_index, len(self._dataset)-1))
        self._get_row_infos()

    def play_audio(self):
        """Play the current audio using sounddevice."""
        try:
            sd.play(self.current_row["y"], self.current_row["sr"])
        except Exception as e:
            return
        
    # -- Private Methods (helpers) --

    # Dataset Methods
    def _get_current_dataset_path(self) -> str:
        """
        Get the most recently edited dataset from the CSV folder

        Returns:
            str: the path for the current dataset 
        """
        subpath = os.path.join(CSV_FOLDER, self._subpath)
        datasets = os.listdir(subpath)
        csv_files = [f for f in datasets if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No Dataset Found!")
        
        latest_file = max(csv_files, key=lambda f: os.path.getctime(os.path.join(subpath, f)))
        return os.path.join(subpath, latest_file)
    
    def _load_current_dataset(self, dataset_path: str = None):
        """
        Loads the newest dataset

        Args:
            dataset_path (str): The optional path to the dataset
        """
        path = self._get_current_dataset_path() if dataset_path == None else dataset_path
        self._dataset = pd.read_csv(path, index_col="track_id")

        # set new columns if not set:
        new_cols = {
            "checked": False,
            "is_voiced": False,
            "multiple_voices": False,
            "interview": False,
            "voice_quality": 0,
            "vocal_content_length_s": 0.0,
        }
        for col, default in new_cols.items():
            if col not in self._dataset.columns:
                self._dataset[col] = default

    def _generate_new_dataset_path(self) -> str:
        """Generates a new name and path for the dataset
        
        Returns:
            str: the path for the new dataset 
        """
        self._get_trimmed_audio
        return os.path.join(CSV_FOLDER, self._subpath, f"dataset_{datetime.now().strftime('%y%m%d_%H%M%S')}.csv")
    
    # Row Methods
    def _get_trimmed_audio(self, audiopath: str) -> Tuple[np.ndarray, int]:
        """Load audio from path and remove silent parts.

        Args:
            audiopath (str): Path to the audio file

        Returns:
            tuple: (sample_array, sample_rate)
        """
        y, sr = librosa.load(audiopath, mono=True)
        intervals = librosa.effects.split(y)
        trimmed_parts = []
        for start, end in intervals:
            trimmed_parts.append(y[..., start:end])
        return np.concatenate(trimmed_parts, axis=-1), sr
    
    def _get_row_infos(self):
        """get the informations of the current row (by self.current_index)
        Stores them as a dict class variable 
        """
        row = self._dataset.iloc[self.current_index]
        y, sr = self._get_trimmed_audio(row.vocal_path)

        self.current_row = {
            **row,
            "track_id": row.name,
            "y": y,
            "sr": sr,
            "vocal_content_length_s": float(y.shape[0]) / float(sr)
        }
    
    def _step_forward(self) -> bool:
        """Steps to the first unchecked dataset row, if there are any

        Returns:
            bool: returns false if all rows have been set to checked
        """
        unchecked_indices = self._dataset.index[~self._dataset['checked']].tolist()
        if not unchecked_indices:
            return False
        self.current_index = self._dataset.index.get_loc(unchecked_indices[0])
        self._get_row_infos()
        return True

    def _insert_row_infox(self):
        """sets the infos from the current row into the dataset row at the current index position
        """
        track_id = self._dataset.iloc[self.current_index].name
        self._dataset.loc[track_id, "checked"] = True
        self._dataset.loc[track_id, "voice_quality"] = self.current_row["voice_quality"]
        self._dataset.loc[track_id, "is_voiced"] = (self.current_row["voice_quality"] == 0)
        self._dataset.loc[track_id, "multiple_voices"] = self.current_row["multiple_voices"]
        self._dataset.loc[track_id, "vocal_content_length_s"] = self.current_row["vocal_content_length_s"]
        self._dataset.loc[track_id, "interview"] = self.current_row["interview"]