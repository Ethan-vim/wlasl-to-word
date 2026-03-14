"""
Episodic batch sampler for N-way K-shot few-shot training.

Generates episodes where each episode contains:
- N randomly chosen classes
- K support + Q query samples per class

Used with ``torch.utils.data.DataLoader`` via the ``batch_sampler`` argument.
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class EpisodicBatchSampler:
    """Yields batches of indices for N-way K-shot + Q-query episodes.

    Parameters
    ----------
    labels : array-like
        Integer class labels for every sample in the dataset.
    n_way : int
        Number of classes per episode.
    k_shot : int
        Number of support samples per class.
    q_query : int
        Number of query samples per class.
    num_episodes : int
        Number of episodes per epoch.
    seed : int or None
        Random seed for reproducibility within an epoch.
    """

    def __init__(
        self,
        labels: np.ndarray,
        n_way: int = 20,
        k_shot: int = 3,
        q_query: int = 2,
        num_episodes: int = 500,
        seed: int | None = None,
    ) -> None:
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.rng = np.random.RandomState(seed)

        # Build class → sample index mapping
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[int(label)].append(idx)

        # Filter classes with enough samples
        min_samples = k_shot + q_query
        self.eligible_classes = [
            c
            for c, indices in self.class_to_indices.items()
            if len(indices) >= min_samples
        ]

        if len(self.eligible_classes) < n_way:
            logger.warning(
                "Only %d classes have >= %d samples (need %d for %d-way). "
                "Reducing n_way to %d.",
                len(self.eligible_classes),
                min_samples,
                n_way,
                n_way,
                len(self.eligible_classes),
            )
            self.n_way = len(self.eligible_classes)

        total = sum(len(self.class_to_indices[c]) for c in self.eligible_classes)
        logger.info(
            "EpisodicBatchSampler: %d eligible classes, %d samples, "
            "%d-way %d-shot %d-query, %d episodes/epoch",
            len(self.eligible_classes),
            total,
            self.n_way,
            self.k_shot,
            self.q_query,
            self.num_episodes,
        )

    def __len__(self) -> int:
        return self.num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            # Sample N classes
            chosen_classes = self.rng.choice(
                self.eligible_classes, size=self.n_way, replace=False
            )

            batch_indices = []
            for c in chosen_classes:
                pool = self.class_to_indices[c]
                selected = self.rng.choice(
                    pool, size=self.k_shot + self.q_query, replace=False
                )
                batch_indices.extend(selected.tolist())

            yield batch_indices
