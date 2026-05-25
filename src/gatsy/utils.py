"""
Copied and adapted from:
https://github.com/difra100/GATSY-Music_Artist_Similarity/blob/main/reduced_data_experiments/utils.py

For more information see model.py
"""

import torch


def save_model(checkpoint, path):
    """
    saves the model to a given path
    """
    torch.save(checkpoint, path)


def load_model(path, model, device):
    """
    loads the model from a given path
    """
    checkpoint = torch.load(path, map_location=device)
    model.GATSY.load_state_dict(checkpoint["modelState"])
    return checkpoint
