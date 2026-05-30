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
    import seaborn as sns

    from src.globals import (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        MODEL_FOLDER,
        STEMS_FOLDER,
        TRACKS_PATH,
        UVR_MODEL_PATH,
    )

    return CSV_FOLDER, DATASET_FOLDER, MODEL_FOLDER, mo, os, pd


@app.cell
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
def _(DATASET_FOLDER, os, pd):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey")

    def parse_js_date(series):
        cleaned = series.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
        return pd.to_datetime(cleaned, format="%a %b %d %Y %H:%M:%S GMT%z")

    participants = pd.read_csv(os.path.join(SURVEY_FOLDER, "participants.csv"), index_col="_id")
    surveyQuestions = pd.read_csv(os.path.join(SURVEY_FOLDER, "surveyQuestions.csv"), index_col="_id")
    surveyAnswers_ = pd.read_csv(os.path.join(SURVEY_FOLDER, "surveyAnswers.csv"), index_col="_id")
    songs = pd.read_csv(os.path.join(SURVEY_FOLDER, "songs.csv"), index_col="_id")
    participants["editDate"] = parse_js_date(participants["editDate"])
    participants["createDate"] = parse_js_date(participants["createDate"])
    participants[participants.surveyCompleted]
    participants["completionTime"] = participants["editDate"] - participants["createDate"]
    participants["completionMinutes"] = participants["completionTime"].dt.total_seconds() / 60
    participants[participants.surveyCompleted]
    surveyAnswers_
    return surveyAnswers_, surveyQuestions


@app.cell
def _(CSV_FOLDER, os, pd):
    full_dataset = pd.read_csv(
        os.path.join(CSV_FOLDER, "LargeDataset", "dataset_all_artist_tracks.csv"),
        index_col="track_id",
    )
    full_dataset
    return (full_dataset,)


@app.cell
def _(full_dataset, track_df):
    track_df["artist_id"] = track_df.apply(lambda x: full_dataset.loc[x.name].artist_id, axis=1)
    return


@app.cell
def _(track_df):
    track_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Link Survey Answers to TrackIDs
    """)
    return


@app.cell
def _(pd, surveyAnswers_, surveyQuestions, track_df):
    def getTrackIdsForAnswer(row):
        questionnaireID = row["questionID"]
        answerKey1 = row["answer_1"]
        answerKey2 = row["answer_2"]
        question = surveyQuestions.loc[questionnaireID]

        def getArtistID(id):
            if question["skip"]:
                return None
            return track_df.loc[id].artist_id

        return pd.Series(
            {
                "track_id_X": question["X"],
                "artist_id_X": getArtistID(question["X"]),
                "track_id_1": question[answerKey1],
                "artist_id_1": getArtistID(question[answerKey1]),
                "track_id_2": question[answerKey2],
                "artist_id_2": getArtistID(question[answerKey2]),
                "skipped": question["skip"],
            }
        )

    surveyAnswers_[
        [
            "track_id_X",
            "artist_id_X",
            "track_id_1",
            "artist_id_1",
            "track_id_2",
            "artist_id_2",
            "skipped",
        ]
    ] = surveyAnswers_.apply(getTrackIdsForAnswer, axis=1)
    surveyAnswers = surveyAnswers_[~surveyAnswers_.skipped].drop(columns="skipped")
    surveyAnswers
    return (surveyAnswers,)


@app.cell
def _(mo):
    mo.md(r"""
    # Load Embedding Model
    - MelSpectrogramEncoder from [pip](https://speechbrain.readthedocs.io/en/latest/installation.html), thanks to speechbrain
    - Encoder Model from [HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb-mel-spec), thanks to speechbrain

    Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., Subakan, C., Dawalatabad, N., Heba, A., Zhong, J., Chou, J.-C., Yeh, S.-L., Fu, S.-W., Liao, C.-F., Rastorgueva, E., Grondin, F., Aris, W., Na, H., Gao, Y., … Bengio, Y. (2021). SpeechBrain: A General-Purpose Speech Toolkit.
    """)
    return


@app.cell
def _():
    # additional imports
    import torch
    import torch.nn.functional as F
    import torchaudio
    from sklearn.model_selection import train_test_split
    from speechbrain.inference.encoders import MelSpectrogramEncoder
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset

    from src.utils import get_trimmed_audio

    return (
        Adam,
        DataLoader,
        Dataset,
        F,
        MelSpectrogramEncoder,
        get_trimmed_audio,
        torch,
        train_test_split,
    )


@app.cell
def _(get_trimmed_audio, track_df):
    track_df["trimmed_audio"] = track_df.song_path.apply(get_trimmed_audio)
    # full_dataset["trimmed_audio"] = full_dataset.song_path.apply(
    #     lambda x: get_trimmed_audio(x, sr=16_000)
    # )
    return


@app.cell
def _():
    RANDOM_STATE = 1

    BATCH_SIZE = 8
    TRAIN_SIZE = 0.8
    EPOCHS = 10
    LEARNING_RATE = 0.00001
    UNFREEZE_LAYERS = 32
    MARGIN = 1.0
    return (
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        MARGIN,
        RANDOM_STATE,
        TRAIN_SIZE,
        UNFREEZE_LAYERS,
    )


@app.cell
def _(MelSpectrogramEncoder):
    encoder = MelSpectrogramEncoder.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb-mel-spec")
    return (encoder,)


@app.cell
def _(Dataset, RANDOM_STATE, full_dataset):
    class TripletDataset(Dataset):
        def __init__(self, df, artist_track_shuffle=False):
            self.data = df.reset_index(drop=True)
            self.artist_track_shuffle = artist_track_shuffle

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            triplet = self.data.iloc[idx]
            if not self.artist_track_shuffle:
                return (
                    full_dataset.loc[triplet["track_id_X"]]["trimmed_audio"],
                    full_dataset.loc[triplet["track_id_1"]]["trimmed_audio"],
                    full_dataset.loc[triplet["track_id_2"]]["trimmed_audio"],
                )
            else:
                return (
                    self.__get_shuffled_track(triplet["artist_id_X"]),
                    self.__get_shuffled_track(triplet["artist_id_1"]),
                    self.__get_shuffled_track(triplet["artist_id_2"]),
                )

        def __get_shuffled_track(self, artist_id: str):
            shuffled_track = (
                full_dataset[full_dataset.artist_id == artist_id].sample(n=1, random_state=RANDOM_STATE).iloc[0]
            )
            return shuffled_track["trimmed_audio"]

    return (TripletDataset,)


@app.cell
def _(F, torch):
    def collate_triplets(batch):
        def pad_audio(wavs):
            max_len = max(w.shape[-1] for w in wavs)
            return torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in wavs])

        anchors, positives, negatives = zip(*batch)
        return pad_audio(anchors), pad_audio(positives), pad_audio(negatives)

    return (collate_triplets,)


@app.cell
def _(
    BATCH_SIZE,
    DataLoader,
    RANDOM_STATE,
    TRAIN_SIZE,
    TripletDataset,
    collate_triplets,
    surveyAnswers,
    train_test_split,
):
    train_df, test_df = train_test_split(surveyAnswers, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)

    train_dataset = TripletDataset(train_df, artist_track_shuffle=True)
    test_dataset = TripletDataset(test_df, artist_track_shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_triplets,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_triplets,
    )
    return test_loader, train_loader


@app.cell
def _(encoder, torch):
    def encode(audio_batch, model=encoder.hparams.embedding_model):
        normalizer = encoder.hparams.normalizer
        audio_batch = audio_batch.to(encoder.device)
        mel = encoder.mel_spectogram(audio_batch)
        lens = torch.ones(mel.shape[0], device=encoder.device)
        feats = normalizer(torch.transpose(mel, 1, 2), lens)
        return model(feats)
        # return torch.nn.functional.normalize(emb, p=2, dim=1)

    return (encode,)


@app.cell
def _(encode, torch, train_loader):
    with torch.no_grad():
        a, p, n = next(iter(train_loader))
        print("d(a,p):", (encode(a) - encode(p)).norm(dim=1).mean().item())
        print("d(a,n):", (encode(a) - encode(n)).norm(dim=1).mean().item())
    return


@app.cell
def _(
    Adam,
    EPOCHS,
    LEARNING_RATE,
    MARGIN,
    UNFREEZE_LAYERS,
    encode,
    encoder,
    mo,
    test_loader,
    torch,
    train_loader,
):
    def fine_tune(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        margin=MARGIN,
        unfreeze_layers=UNFREEZE_LAYERS,
    ):
        model = encoder.hparams.embedding_model
        for p in model.parameters():
            p.requires_grad = True

        params = list(model.parameters())
        for p in params[:-unfreeze_layers]:
            p.requires_grad = False

        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)
        model.train()

        for epoch in mo.status.progress_bar(range(epochs)):
            total_loss = 0.0
            for anchor_mel, positive_mel, negative_mel in mo.status.progress_bar(train_loader):
                optimizer.zero_grad()

                emb_anchor = encode(anchor_mel, model)
                emb_positive = encode(positive_mel, model)
                emb_negative = encode(negative_mel, model)

                loss = triplet_loss_fn(emb_anchor, emb_positive, emb_negative)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                test_loss = sum(
                    triplet_loss_fn(encode(a, model), encode(p, model), encode(n, model)).item()
                    for a, p, n in test_loader
                ) / len(test_loader)
                a, p, n = next(iter(train_loader))
                da = (encode(a, model) - encode(p, model)).norm(dim=1).mean()
                dn = (encode(a, model) - encode(n, model)).norm(dim=1).mean()
            print(
                f"Epoch {epoch + 1}/{epochs} | Train loss: {total_loss / len(train_loader):.4f} | Test loss: {test_loss:.4f} | d(a,p): {da:.4f} | d(a,n): {dn:.4f} | gap: {dn - da:.4f}"
            )
            model.train()

        return encoder

    return (fine_tune,)


@app.cell
def _(fine_tune):
    fine_tune()
    return


@app.cell
def _(EPOCHS, LEARNING_RATE, MARGIN, MODEL_FOLDER, UNFREEZE_LAYERS, os):
    finetuned_encoder_path = os.path.join(
        MODEL_FOLDER,
        "finetuned_encoder",
        f"spkrec-ecapa-voxceleb-mel-spec_shuffle_{UNFREEZE_LAYERS}_{MARGIN}_{EPOCHS}_{LEARNING_RATE}.pt",
    )
    return (finetuned_encoder_path,)


@app.cell
def _(encoder, finetuned_encoder_path, torch):
    torch.save(encoder.hparams.embedding_model.state_dict(), finetuned_encoder_path)
    return


@app.cell
def _(
    EPOCHS,
    LEARNING_RATE,
    MARGIN,
    MODEL_FOLDER,
    UNFREEZE_LAYERS,
    encoder,
    fine_tune,
    os,
    torch,
):
    fine_tune()
    torch.save(
        encoder.hparams.embedding_model.state_dict(),
        os.path.join(
            MODEL_FOLDER,
            "finetuned_encoder",
            f"spkrec-ecapa-voxceleb-mel-spec_shuffle_{UNFREEZE_LAYERS}_{MARGIN}_{EPOCHS * 4}_{LEARNING_RATE}.pt",
        ),
    )
    return


if __name__ == "__main__":
    app.run()
