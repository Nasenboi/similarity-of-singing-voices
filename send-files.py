import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from os import getenv, path
    from random import choice

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import paramiko
    from dotenv import load_dotenv

    import utils

    load_dotenv()
    return getenv, mo, paramiko, path, utils


@app.cell
def _(getenv, path):
    RANDOM_STATE = 42
    DATASET_FOLDER = getenv("DATASET_FOLDER")
    METADATA_FOLDER = path.join(DATASET_FOLDER, "fma_metadata")
    AUDIO_FOLDER = path.join(DATASET_FOLDER, "fma_large")
    TRACKS_PATH = path.join(METADATA_FOLDER, "tracks.csv")
    return AUDIO_FOLDER, METADATA_FOLDER, DATASET_FOLDER, TRACKS_PATH


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
def _(AUDIO_FOLDER, tracks_df, utils):
    tracks_df["audio_path"] = [
        utils.get_audio_path(AUDIO_FOLDER, i) for i in tracks_df.index
    ]
    return


@app.cell
def _(path):
    REMOTE_DATASET_FOLDER = "/home/chr1s/Documents/MusicVoiceCluster"
    REMOTE_METADATA_FOLDER = path.join(REMOTE_DATASET_FOLDER, "fma_metadata")
    REMOTE_AUDIO_FOLDER = path.join(REMOTE_DATASET_FOLDER, "fma_large")
    return REMOTE_METADATA_FOLDER, REMOTE_DATASET_FOLDER


@app.cell
def _(paramiko):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("169.1.1.2", username="chr1s", password="8n9D#r8n9D#r", port=222)
    return (ssh,)


@app.cell
def _(METADATA_FOLDER, REMOTE_METADATA_FOLDER, path, ssh):
    sftp = ssh.open_sftp()
    # sftp.mkdir(REMOTE_METADATA_FOLDER)
    sftp.put(
        path.join(METADATA_FOLDER, "tracks_df_embedding_3D.csv"),
        path.join(REMOTE_METADATA_FOLDER, "tracks_df_embedding_3D.csv"),
    )
    return (sftp,)


@app.cell
def _(DATASET_FOLDER, REMOTE_DATASET_FOLDER, path, sftp):
    def copy_audio_to_remote(source: str):
        destination = source.replace(DATASET_FOLDER, REMOTE_DATASET_FOLDER)
        destination_base_path = path.dirname(destination)
        try:
            sftp.stat(destination_base_path)
        except FileNotFoundError:
            sftp.mkdir(destination_base_path)
        sftp.put(source, destination)
    return (copy_audio_to_remote,)


@app.cell
def _(tracks_df):
    tracks_df["audio_path"]
    return


@app.cell
def _(copy_audio_to_remote, mo, sftp, ssh, tracks_df):
    for audio_path in mo.status.progress_bar(
        collection=tracks_df["audio_path"],
        title="Transfer Audio Files",
        show_eta=True,
        show_rate=True,
    ):
        copy_audio_to_remote(audio_path)

    sftp.close()
    ssh.close()
    return


if __name__ == "__main__":
    app.run()
