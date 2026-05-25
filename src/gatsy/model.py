"""
gatsy/model.py file

based on the training notebook provided by Andrea Giuseppe Di Francesco
https://github.com/difra100/GATSY-Music_Artist_Similarity/

For more Information see:
A. G. Di Francesco, G. Giampietro, I. Spinelli and D. Comminiello, "GATSY: Graph Attention Network for Music Artist Similarity,"
2025 International Joint Conference on Neural Networks (IJCNN), Rome, Italy, 2025, pp. 1-8, doi: 10.1109/IJCNN64981.2025.11228629.

Note: Use only in Marimo Notebooks, if you do not have marimo update the progress bar logic!
"""

import os
from datetime import datetime

import marimo as mo
import numpy as np
import torch
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from torch.nn import TripletMarginLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from .architectures import GATSY
from .utils import *
from .utils import save_model


class Trainer:
    """This class contains all the method needed to train, test, and save the model."""

    def __init__(
        self,
        model: GATSY,
        train_loader: DataLoader,
        train_x: np.ndarray,  # edges are computed at start of training
        test_loader: DataLoader,
        lr: float,
        epochs: int,
        margin: float,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_neighbors: int = 5,
        test_x: np.ndarray = None,  # optional, if not given all nodes are used in testing and training
        weight_decay=None,
        model_path=None,
        model_name=None,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.train_x = torch.tensor(train_x).to(self.device)
        self.train_edges = self.__get_edges(train_x, n_neighbors)
        self.test_loader = test_loader
        self.test_x = torch.tensor(test_x).to(self.device) if test_x is not None else self.train_x
        self.test_edges = self.__get_edges(test_x, n_neighbors) if test_x is not None else self.train_edges
        self.epochs = epochs
        self.model_path = model_path if model_path is not None else os.getcwd()
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_name = model_name if model_name is not None else f"gatsy_{epochs}_{lr}_{now}.pt"

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_function = TripletMarginLoss(margin=margin)
        self.first = True
        self.checkpoint = {}

    def train(self):
        """
        Train the number for the given number of epochs
        """

        self.model.train()
        for epoch in mo.status.progress_bar(range(self.epochs), title="Running Training Loop"):
            self.optimizer.zero_grad()

            out = self.model(self.train_x, self.train_edges)
            total_loss = torch.tensor(0.0, device=self.device)
            for a, p, n in mo.status.progress_bar(
                self.train_loader, remove_on_exit=True, title=f"Running Epoch {epoch}/{self.epochs}"
            ):
                a_emb, p_emb, n_emb = out[a], out[p], out[n]
                total_loss += self.loss_function(a_emb, p_emb, n_emb)

            total_loss.backward()
            self.optimizer.step()

            self.loss_train_f = total_loss.item() / len(self.train_loader.dataset)
            self.__end_epoch()

        self.checkpoint["modelState"] = self.model.state_dict()
        save_model(self.checkpoint, os.path.join(self.model_path, self.model_name))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = torch.tensor(0.0, device=self.device)
            acc_test_list = []
            pred_acc_test_list = []
            out = self.model(self.test_x, self.test_edges)
            for a, p, n in mo.status.progress_bar(
                self.test_loader, remove_on_exit=True, title=f"Running Test on {len(self.test_loader)} Triplets"
            ):
                a_emb, p_emb, n_emb = out[a], out[p], out[n]
                total_loss += self.loss_function(a_emb, p_emb, n_emb)

                score, pred = self.__get_triplet_acc((a_emb, p_emb, n_emb))
                acc_test_list.append(score)
                pred_acc_test_list.append(pred)

        self.loss_test_f = total_loss.item() / len(self.test_loader.dataset)
        self.accuracy_score = sum(acc_test_list) / len(acc_test_list)
        self.pred_accuracy = sum(pred_acc_test_list) / len(pred_acc_test_list)
        self.model.train()

    def __get_triplet_acc(self, triplet):
        a_emb = triplet[0].cpu().numpy()
        p_emb = triplet[1].cpu().numpy()
        n_emb = triplet[2].cpu().numpy()

        a_emb = a_emb.mean(axis=0)
        p_emb = p_emb.mean(axis=0)
        n_emb = n_emb.mean(axis=0)

        dist_ap = euclidean(a_emb, p_emb)
        dist_an = euclidean(a_emb, n_emb)
        score = 1.0 if dist_an > dist_ap else 0.0
        pred_acc = (dist_an - dist_ap) / (dist_ap + dist_an)
        return score, pred_acc

    def __end_epoch(self):
        """
        This method is called at the end of each epoch and its purpose is to save the model state, and metrics, in order to be loaded again when needed.
        """
        self.test()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if self.first:
            self.checkpoint["loss_train"] = [self.loss_train_f]
            self.checkpoint["loss_test"] = [self.loss_test_f]
            self.checkpoint["accuracy_score"] = [self.accuracy_score]
            self.checkpoint["pred_accuracy"] = [self.pred_accuracy]
            self.first = False
        else:
            self.checkpoint["loss_train"] += [self.loss_train_f]
            self.checkpoint["loss_test"] += [self.loss_test_f]
            self.checkpoint["accuracy_score"] += [self.accuracy_score]
            self.checkpoint["pred_accuracy"] += [self.pred_accuracy]

    def __get_edges(self, x: np.ndarray, n_neighbors: int) -> torch.Tensor:
        """
        Build k-NN graph from feature matrix.
        Args:
            x: node features, shape (n_nodes, n_dims)
            n_neighbors: number of nearest neighbors per node (excluding self)
        Returns:
            edge_index: COO tensor, shape (2, num_edges), dtype torch.long
        """
        n_nodes = x.shape[0]
        k = min(n_neighbors, n_nodes - 1)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute")
        nbrs.fit(x)
        knn_indices = nbrs.kneighbors(x, return_distance=False)
        knn_indices = knn_indices[:, 1:]

        src = []
        dst = []
        for i in range(n_nodes):
            for j in knn_indices[i]:
                src.append(i)
                dst.append(j)
                src.append(j)
                dst.append(i)

        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index
