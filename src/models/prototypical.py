"""
Prototypical Network for few-shot sign language recognition.

Wraps an encoder (STGCNEncoder) and performs metric-learning-based
classification: during training, episodic N-way K-shot tasks compute
distances to class prototypes; at inference, stored prototypes from the
full training set are used for nearest-prototype classification.

Reference: Snell et al., "Prototypical Networks for Few-shot Learning",
NeurIPS 2017.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.stgcn import STGCNEncoder, build_stgcn_encoder

logger = logging.getLogger(__name__)


class PrototypicalNetwork(nn.Module):
    """Prototypical Network with ST-GCN encoder backbone.

    During training (``forward``), accepts support and query sets and
    returns negative squared-Euclidean distances as logits.

    At inference, call ``compute_prototypes`` once with the training data,
    then ``classify`` for each new sample.
    """

    def __init__(self, encoder: STGCNEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        # Registered buffer: class prototypes (set via compute_prototypes)
        self.register_buffer(
            "prototypes", torch.zeros(0), persistent=True
        )
        self._num_classes = 0

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        """Episodic forward pass.

        Parameters
        ----------
        support_x : (N_way * K_shot, T, F)
            Support set samples.
        support_y : (N_way * K_shot,)
            Support labels (0-indexed within the episode, i.e. 0..N_way-1).
        query_x : (N_way * Q_query, T, F)
            Query set samples.

        Returns
        -------
        torch.Tensor
            Negative squared Euclidean distances: ``(N_way * Q_query, N_way)``.
            Apply ``F.cross_entropy(-distances, query_y)`` for the loss.
        """
        # Encode all samples
        all_x = torch.cat([support_x, query_x], dim=0)
        all_emb = self.encoder(all_x)

        n_support = support_x.size(0)
        support_emb = all_emb[:n_support]
        query_emb = all_emb[n_support:]

        # Compute prototypes: mean embedding per class in support set
        n_way = support_y.unique().size(0)
        prototypes = torch.zeros(n_way, self.embedding_dim, device=all_x.device)
        for c in range(n_way):
            mask = support_y == c
            prototypes[c] = support_emb[mask].mean(dim=0)

        # Squared Euclidean distance: (Q, N_way)
        # Manual computation instead of torch.cdist (unsupported backward on MPS)
        distances = (
            query_emb.unsqueeze(1) - prototypes.unsqueeze(0)
        ).pow(2).sum(dim=-1)
        return -distances  # negative distances as logits

    @torch.no_grad()
    def compute_prototypes(self, dataloader: DataLoader) -> None:
        """Compute and store class prototypes from the full training set.

        After calling this, use ``classify()`` for inference.
        """
        self.eval()
        device = next(self.parameters()).device

        embeddings: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            emb = self.encoder(batch_x)
            embeddings.append(emb.cpu())
            labels.append(batch_y)

        all_emb = torch.cat(embeddings, dim=0)
        all_labels = torch.cat(labels, dim=0)

        unique_classes = all_labels.unique(sorted=True)
        self._num_classes = len(unique_classes)
        protos = torch.zeros(self._num_classes, self.embedding_dim)

        for i, c in enumerate(unique_classes):
            mask = all_labels == c
            protos[i] = all_emb[mask].mean(dim=0)
            protos[i] = F.normalize(protos[i], dim=0)

        self.prototypes = protos.to(device)
        logger.info(
            "Computed prototypes for %d classes (embedding_dim=%d)",
            self._num_classes,
            self.embedding_dim,
        )

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classify samples using stored prototypes.

        Parameters
        ----------
        x : (B, T, F)
            Input keypoint sequences.

        Returns
        -------
        torch.Tensor
            Negative squared distances as logits: ``(B, num_classes)``.
        """
        emb = self.encoder(x)  # (B, D)
        distances = (
            emb.unsqueeze(1) - self.prototypes.unsqueeze(0)
        ).pow(2).sum(dim=-1)
        return -distances


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(cfg: Any) -> PrototypicalNetwork:
    """Build a PrototypicalNetwork from a config object."""
    encoder = build_stgcn_encoder(cfg)
    model = PrototypicalNetwork(encoder)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("PrototypicalNetwork: %.2fM trainable parameters", total / 1e6)
    return model
