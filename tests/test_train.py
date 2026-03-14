"""Tests for src.training.train_prototypical — episode splitting, training helpers."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.train_prototypical import _split_episode


# ---------------------------------------------------------------------------
# _split_episode
# ---------------------------------------------------------------------------


class TestSplitEpisode:
    def test_output_shapes(self):
        n_way, k_shot, q_query = 3, 2, 1
        per_class = k_shot + q_query
        total = n_way * per_class
        x = torch.randn(total, 16, 100)
        y = torch.zeros(total, dtype=torch.long)  # labels don't matter for shape

        support_x, support_y, query_x, query_y = _split_episode(
            x, y, n_way, k_shot, q_query
        )
        assert support_x.shape == (n_way * k_shot, 16, 100)
        assert query_x.shape == (n_way * q_query, 16, 100)
        assert support_y.shape == (n_way * k_shot,)
        assert query_y.shape == (n_way * q_query,)

    def test_local_labels(self):
        n_way, k_shot, q_query = 4, 2, 1
        per_class = k_shot + q_query
        total = n_way * per_class
        x = torch.randn(total, 8, 50)
        y = torch.zeros(total, dtype=torch.long)

        support_x, support_y, query_x, query_y = _split_episode(
            x, y, n_way, k_shot, q_query
        )
        # Labels should be 0..n_way-1
        assert support_y.max().item() == n_way - 1
        assert support_y.min().item() == 0
        assert query_y.max().item() == n_way - 1
        assert query_y.min().item() == 0

    def test_support_query_disjoint(self):
        """Support and query should contain different samples."""
        n_way, k_shot, q_query = 2, 3, 2
        per_class = k_shot + q_query
        total = n_way * per_class
        # Use unique values so we can verify disjointness
        x = torch.arange(total).float().unsqueeze(1).unsqueeze(2)
        y = torch.zeros(total, dtype=torch.long)

        support_x, _, query_x, _ = _split_episode(
            x, y, n_way, k_shot, q_query
        )
        support_ids = set(support_x.squeeze().tolist())
        query_ids = set(query_x.squeeze().tolist())
        assert len(support_ids & query_ids) == 0

    def test_each_class_has_k_support(self):
        n_way, k_shot, q_query = 3, 4, 2
        per_class = k_shot + q_query
        total = n_way * per_class
        x = torch.randn(total, 8, 50)
        y = torch.zeros(total, dtype=torch.long)

        _, support_y, _, query_y = _split_episode(
            x, y, n_way, k_shot, q_query
        )
        for c in range(n_way):
            assert (support_y == c).sum().item() == k_shot
            assert (query_y == c).sum().item() == q_query
