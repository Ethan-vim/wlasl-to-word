"""
Spatiotemporal Graph Convolutional Network (ST-GCN) encoder for skeleton-based
sign language recognition.

Processes MediaPipe Holistic keypoints as a graph where joints are nodes and
bone connections are edges.  Body, left hand, and right hand are processed
as separate graph branches, then merged into a unified embedding.

Reference: Yan et al., "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition", AAAI 2018.
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe Holistic skeleton connections
# ---------------------------------------------------------------------------

# Body pose: 33 landmarks (indices 0-32 in the full 543-keypoint array)
BODY_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),                        # left eye → ear
    (0, 4), (4, 5), (5, 6), (6, 8),                        # right eye → ear
    (9, 10),                                                 # mouth
    (11, 12),                                                # shoulders
    (11, 13), (13, 15),                                      # left arm
    (12, 14), (14, 16),                                      # right arm
    (15, 17), (15, 19), (15, 21),                            # left wrist → fingers
    (16, 18), (16, 20), (16, 22),                            # right wrist → fingers
    (11, 23), (12, 24),                                      # torso
    (23, 24),                                                # hips
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),       # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),       # right leg
]
BODY_NUM_JOINTS = 33

# Hand: 21 landmarks (left: indices 33-53, right: 54-74)
HAND_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),                         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                         # index
    (0, 9), (9, 10), (10, 11), (11, 12),                    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),                  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),                  # pinky
    (5, 9), (9, 13), (13, 17),                               # palm
]
HAND_NUM_JOINTS = 21


# ---------------------------------------------------------------------------
# Graph construction utilities
# ---------------------------------------------------------------------------


def _build_adjacency(edges: list[tuple[int, int]], num_nodes: int) -> np.ndarray:
    """Build binary adjacency matrix from an undirected edge list."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def _normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Row-normalize adjacency: D^{-1} A."""
    D = A.sum(axis=1, keepdims=True)
    D = np.where(D > 0, D, 1.0)
    return A / D


def build_spatial_graph(
    edges: list[tuple[int, int]], num_nodes: int
) -> np.ndarray:
    """Build partitioned adjacency for spatial graph convolution.

    Returns shape ``(2, V, V)``  — partition 0 is the identity (self-loop),
    partition 1 is the row-normalized neighbor adjacency.
    """
    A = _build_adjacency(edges, num_nodes)
    A_self = np.eye(num_nodes, dtype=np.float32)
    A_neighbor = _normalize_adjacency(A)
    return np.stack([A_self, A_neighbor])  # (2, V, V)


# ---------------------------------------------------------------------------
# ST-GCN layers
# ---------------------------------------------------------------------------


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution with learnable edge importance."""

    def __init__(
        self, in_channels: int, out_channels: int, adj: np.ndarray
    ) -> None:
        super().__init__()
        K = adj.shape[0]  # number of adjacency partitions
        self.K = K
        self.register_buffer("adj", torch.from_numpy(adj))
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones(adj.shape[1], adj.shape[2])) for _ in range(K)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, V)`` → ``(B, C_out, T, V)``."""
        B, C_in, T, V = x.shape
        x_conv = self.conv(x)  # (B, C_out*K, T, V)
        C_out = x_conv.shape[1] // self.K
        x_conv = x_conv.view(B, self.K, C_out, T, V)

        out = torch.zeros(B, C_out, T, V, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            A_k = self.adj[k] * self.edge_importance[k]
            # Use matmul instead of einsum — einsum hangs on MPS backend
            out = out + torch.matmul(x_conv[:, k], A_k.t())
        return out


class STGCNBlock(nn.Module):
    """Single spatiotemporal block: spatial GCN → temporal conv → residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj: np.ndarray,
        stride: int = 1,
        temporal_kernel: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, adj)
        self.bn_s = nn.BatchNorm2d(out_channels)

        padding = (temporal_kernel - 1) // 2
        self.tcn = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1),
            stride=(stride, 1),
            padding=(padding, 0),
        )
        self.bn_t = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, V)`` → ``(B, C_out, T', V)``."""
        res = self.residual(x)
        x = F.relu(self.bn_s(self.gcn(x)))
        x = self.dropout(F.relu(self.bn_t(self.tcn(x))))
        return x + res


class STGCNBranch(nn.Module):
    """Stack of ST-GCN blocks for a single skeleton partition."""

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        adj: np.ndarray,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in channels:
            layers.append(STGCNBlock(ch_in, ch_out, adj, dropout=dropout))
            ch_in = ch_out
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, V)`` → ``(B, out_channels)`` via global average pool."""
        x = self.layers(x)  # (B, C_out, T, V)
        return x.mean(dim=[2, 3])  # (B, C_out)


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------


class STGCNEncoder(nn.Module):
    """Partitioned ST-GCN encoder for MediaPipe Holistic keypoints.

    Processes body (33 joints), left hand (21 joints), and right hand
    (21 joints) through separate ST-GCN branches, then merges into a
    single L2-normalized embedding.

    Input:  ``(B, T, num_keypoints * C)``  where C = 3 or 6
    Output: ``(B, embedding_dim)``  L2-normalized
    """

    def __init__(
        self,
        num_keypoints: int = 543,
        embedding_dim: int = 128,
        channels: list[int] | None = None,
        dropout: float = 0.1,
        use_motion: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.use_motion = use_motion
        self.embedding_dim = embedding_dim

        in_channels = 6 if use_motion else 3
        if channels is None:
            channels = [64, 128, 128]

        body_adj = build_spatial_graph(BODY_EDGES, BODY_NUM_JOINTS)
        hand_adj = build_spatial_graph(HAND_EDGES, HAND_NUM_JOINTS)

        self.body_branch = STGCNBranch(in_channels, channels, body_adj, dropout)
        self.lhand_branch = STGCNBranch(in_channels, channels, hand_adj, dropout)
        self.rhand_branch = STGCNBranch(in_channels, channels, hand_adj, dropout)

        branch_out = channels[-1]
        self.projection = nn.Sequential(
            nn.Linear(3 * branch_out, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode keypoint sequence to L2-normalized embedding.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, num_keypoints * C)`` where C = 3 or 6.

        Returns
        -------
        torch.Tensor
            Shape ``(B, embedding_dim)``, L2-normalized.
        """
        B, T, _ = x.shape
        C = 6 if self.use_motion else 3
        x = x.view(B, T, self.num_keypoints, C)

        # Extract partitions and permute to (B, C, T, V)
        body = x[:, :, :33, :].permute(0, 3, 1, 2).contiguous()
        lhand = x[:, :, 33:54, :].permute(0, 3, 1, 2).contiguous()
        rhand = x[:, :, 54:75, :].permute(0, 3, 1, 2).contiguous()

        body_feat = self.body_branch(body)
        lhand_feat = self.lhand_branch(lhand)
        rhand_feat = self.rhand_branch(rhand)

        merged = torch.cat([body_feat, lhand_feat, rhand_feat], dim=1)
        embedding = self.projection(merged)
        return F.normalize(embedding, dim=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_stgcn_encoder(cfg: Any) -> STGCNEncoder:
    """Build an STGCNEncoder from a configuration object."""
    encoder = STGCNEncoder(
        num_keypoints=getattr(cfg, "num_keypoints", 543),
        embedding_dim=getattr(cfg, "embedding_dim", 128),
        channels=getattr(cfg, "gcn_channels", None),
        dropout=getattr(cfg, "dropout", 0.1),
        use_motion=getattr(cfg, "use_motion", False),
    )
    param_count = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info("Built STGCNEncoder with %.2fM parameters", param_count / 1e6)
    return encoder
