"""Tests for src.models — STGCNEncoder, PrototypicalNetwork, build functions."""

import pytest
import torch

from src.models.stgcn import (
    STGCNEncoder,
    STGCNBlock,
    STGCNBranch,
    SpatialGraphConv,
    build_spatial_graph,
    build_stgcn_encoder,
    BODY_EDGES,
    BODY_NUM_JOINTS,
    HAND_EDGES,
    HAND_NUM_JOINTS,
)
from src.models.prototypical import PrototypicalNetwork, build_model
from src.training.config import Config


NUM_KP = 543
B, T = 2, 16


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestBuildSpatialGraph:
    def test_output_shape(self):
        adj = build_spatial_graph(BODY_EDGES, BODY_NUM_JOINTS)
        assert adj.shape == (2, BODY_NUM_JOINTS, BODY_NUM_JOINTS)

    def test_identity_partition(self):
        adj = build_spatial_graph(HAND_EDGES, HAND_NUM_JOINTS)
        # Partition 0 should be identity
        for i in range(HAND_NUM_JOINTS):
            assert adj[0, i, i] == 1.0

    def test_neighbor_partition_symmetric(self):
        adj = build_spatial_graph(BODY_EDGES, BODY_NUM_JOINTS)
        # Adjacency should have non-zero entries for connected joints
        # Shoulders (11, 12) are connected
        assert adj[1, 11, 12] > 0 or adj[1, 12, 11] > 0


# ---------------------------------------------------------------------------
# SpatialGraphConv
# ---------------------------------------------------------------------------


class TestSpatialGraphConv:
    def test_forward_shape(self):
        adj = build_spatial_graph(HAND_EDGES, HAND_NUM_JOINTS)
        conv = SpatialGraphConv(3, 64, adj)
        x = torch.randn(B, 3, T, HAND_NUM_JOINTS)
        out = conv(x)
        assert out.shape == (B, 64, T, HAND_NUM_JOINTS)


# ---------------------------------------------------------------------------
# STGCNBlock
# ---------------------------------------------------------------------------


class TestSTGCNBlock:
    def test_forward_shape(self):
        adj = build_spatial_graph(BODY_EDGES, BODY_NUM_JOINTS)
        block = STGCNBlock(3, 64, adj, dropout=0.0)
        x = torch.randn(B, 3, T, BODY_NUM_JOINTS)
        out = block(x)
        assert out.shape == (B, 64, T, BODY_NUM_JOINTS)

    def test_residual_channel_change(self):
        adj = build_spatial_graph(HAND_EDGES, HAND_NUM_JOINTS)
        block = STGCNBlock(3, 128, adj, dropout=0.0)
        x = torch.randn(B, 3, T, HAND_NUM_JOINTS)
        out = block(x)
        assert out.shape == (B, 128, T, HAND_NUM_JOINTS)


# ---------------------------------------------------------------------------
# STGCNEncoder
# ---------------------------------------------------------------------------


class TestSTGCNEncoder:
    def test_forward_shape(self):
        encoder = STGCNEncoder(
            num_keypoints=NUM_KP, embedding_dim=128,
            channels=[64, 128], dropout=0.0, use_motion=False,
        )
        x = torch.randn(B, T, NUM_KP * 3)
        out = encoder(x)
        assert out.shape == (B, 128)

    def test_forward_with_motion(self):
        encoder = STGCNEncoder(
            num_keypoints=NUM_KP, embedding_dim=128,
            channels=[64, 128], dropout=0.0, use_motion=True,
        )
        x = torch.randn(B, T, NUM_KP * 6)
        out = encoder(x)
        assert out.shape == (B, 128)

    def test_output_l2_normalized(self):
        encoder = STGCNEncoder(
            num_keypoints=NUM_KP, embedding_dim=64,
            channels=[32, 64], dropout=0.0,
        )
        encoder.eval()
        x = torch.randn(B, T, NUM_KP * 3)
        with torch.no_grad():
            out = encoder(x)
        norms = out.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(B), atol=1e-5, rtol=0)

    def test_build_from_config(self):
        cfg = Config(
            wlasl_variant=10,
            num_keypoints=NUM_KP,
            d_model=64,
            gcn_channels=[32, 64],
            dropout=0.0,
            use_motion=False,
        )
        encoder = build_stgcn_encoder(cfg)
        assert isinstance(encoder, STGCNEncoder)
        x = torch.randn(1, T, NUM_KP * 3)
        assert encoder(x).shape == (1, 64)


# ---------------------------------------------------------------------------
# PrototypicalNetwork
# ---------------------------------------------------------------------------


class TestPrototypicalNetwork:
    def _make_model(self, embedding_dim=64, use_motion=False):
        encoder = STGCNEncoder(
            num_keypoints=NUM_KP, embedding_dim=embedding_dim,
            channels=[32, 64], dropout=0.0, use_motion=use_motion,
        )
        return PrototypicalNetwork(encoder)

    def test_forward_episodic(self):
        model = self._make_model()
        n_way, k_shot, q_query = 3, 2, 1
        support_x = torch.randn(n_way * k_shot, T, NUM_KP * 3)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(n_way * q_query, T, NUM_KP * 3)
        logits = model(support_x, support_y, query_x)
        assert logits.shape == (n_way * q_query, n_way)

    def test_classify_with_prototypes(self):
        model = self._make_model()
        # Manually set prototypes
        model.prototypes = torch.randn(5, 64)
        model._num_classes = 5
        x = torch.randn(B, T, NUM_KP * 3)
        logits = model.classify(x)
        assert logits.shape == (B, 5)

    def test_compute_prototypes(self):
        model = self._make_model()
        # Create simple dataset
        data = torch.randn(8, T, NUM_KP * 3)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(data, labels)
        loader = DataLoader(ds, batch_size=4)
        model.compute_prototypes(loader)
        assert model.prototypes.shape == (4, 64)

    def test_model_is_differentiable(self):
        model = self._make_model()
        n_way, k_shot, q_query = 2, 2, 1
        support_x = torch.randn(n_way * k_shot, T, NUM_KP * 3)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(n_way * q_query, T, NUM_KP * 3)
        logits = model(support_x, support_y, query_x)
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_build_from_config(self):
        cfg = Config(
            wlasl_variant=10,
            num_keypoints=NUM_KP,
            d_model=64,
            gcn_channels=[32, 64],
            dropout=0.0,
            use_motion=False,
        )
        model = build_model(cfg)
        assert isinstance(model, PrototypicalNetwork)
        x = torch.randn(1, T, NUM_KP * 3)
        # Need prototypes for classify
        model.prototypes = torch.randn(10, 64)
        model._num_classes = 10
        logits = model.classify(x)
        assert logits.shape == (1, 10)

    def test_build_with_motion(self):
        cfg = Config(
            wlasl_variant=10,
            num_keypoints=NUM_KP,
            d_model=64,
            gcn_channels=[32, 64],
            dropout=0.0,
            use_motion=True,
        )
        model = build_model(cfg)
        x = torch.randn(1, T, NUM_KP * 6)
        model.prototypes = torch.randn(10, 64)
        model._num_classes = 10
        logits = model.classify(x)
        assert logits.shape == (1, 10)
