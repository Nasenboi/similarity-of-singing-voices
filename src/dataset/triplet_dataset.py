from torch.utils.data import Dataset
import torch


class TripletDataset(Dataset):
    def __init__(
        self,
        triplet_df,
        node_df,
        node_id_key,
        dtype=torch.float,
    ):
        self.triplet_data = triplet_df
        self.node_data = node_df
        self.node_id_key = node_id_key
        self.dtype = dtype

    def __len__(self):
        return len(self.triplet_data)

    def __getitem__(self, idx):
        triplet = self.triplet_data.iloc[idx]
        return (
            self.node_data.loc[triplet["track_id_X"]][self.node_id_key],
            self.node_data.loc[triplet["track_id_1"]][self.node_id_key],
            self.node_data.loc[triplet["track_id_2"]][self.node_id_key],
        )
