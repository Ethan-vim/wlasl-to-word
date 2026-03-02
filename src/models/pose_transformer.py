"""
Keypoint-based models for sign language recognition.

Provides two architectures for Approach A:

- ``PoseTransformer``: Transformer encoder with learnable positional encoding.
- ``PoseBiLSTM``: Bidirectional 2-layer LSTM alternative.

Both share the same forward interface: input ``(B, T, input_dim)``
and output ``(B, num_classes)`` logits, where ``input_dim`` is
``num_keypoints * 3`` (position only) or ``num_keypoints * 6``
(position + velocity when ``use_motion=True``).
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PoseTransformer(nn.Module):
    """Transformer encoder for keypoint-sequence classification.

    Architecture:
        Linear projection -> Learnable positional encoding ->
        N x TransformerEncoderLayer (pre-norm) ->
        Global average pooling -> Dropout -> LayerNorm -> Linear head

    Parameters
    ----------
    num_keypoints : int
        Number of keypoints per frame (e.g., 543 for full MediaPipe Holistic).
    num_classes : int
        Number of output classes (glosses).
    d_model : int
        Transformer embedding dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    dropout : float
        Dropout rate.
    T : int
        Maximum sequence length (for positional encoding).
    dim_feedforward : int or None
        Feed-forward dimension inside the Transformer.  Defaults to ``4 * d_model``.
    """

    def __init__(
        self,
        num_keypoints: int = 543,
        num_classes: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3,
        T: int = 64,
        dim_feedforward: int | None = None,
        use_motion: bool = False,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.T = T
        self.use_motion = use_motion

        features_per_kp = 6 if use_motion else 3
        input_dim = num_keypoints * features_per_kp
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, T, d_model) * 0.02)

        # Transformer encoder with pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input keypoints of shape ``(B, T, num_keypoints * 3)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        B, T, _ = x.shape

        # Project input features
        x = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding (handle variable lengths up to self.T)
        x = x + self.pos_encoding[:, :T, :]

        # Transformer encoder
        x = self.transformer(x)  # (B, T, d_model)

        # Global average pooling over time
        x = x.mean(dim=1)  # (B, d_model)

        # Classification
        logits = self.head(x)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings (before the classification head).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, num_keypoints * 3)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, d_model)``.
        """
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :T, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x


class PoseBiLSTM(nn.Module):
    """Bidirectional LSTM for keypoint-sequence classification.

    Architecture:
        Linear projection -> 2-layer BiLSTM -> Global average pooling ->
        Dropout -> LayerNorm -> Linear head

    Parameters
    ----------
    num_keypoints : int
        Number of keypoints per frame.
    num_classes : int
        Number of output classes.
    d_model : int
        Hidden dimension of the LSTM (each direction uses d_model // 2).
    num_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    T : int
        Maximum sequence length (unused, kept for API compatibility).
    """

    def __init__(
        self,
        num_keypoints: int = 543,
        num_classes: int = 100,
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        T: int = 64,
        use_motion: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.use_motion = use_motion

        features_per_kp = 6 if use_motion else 3
        input_dim = num_keypoints * features_per_kp
        hidden_dim = d_model // 2  # each direction produces hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head (BiLSTM output dim = 2 * hidden_dim = d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input keypoints of shape ``(B, T, num_keypoints * 3)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        x = self.input_proj(x)  # (B, T, d_model)
        x, _ = self.lstm(x)  # (B, T, d_model)

        # Global average pooling over time
        x = x.mean(dim=1)  # (B, d_model)

        logits = self.head(x)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before the classification head.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, num_keypoints * 3)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, d_model)``.
        """
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_pose_model(cfg: Any) -> nn.Module:
    """Build a pose-based model from a configuration object.

    Parameters
    ----------
    cfg : object
        Configuration with attributes: ``approach``, ``num_keypoints``,
        ``num_classes``, ``d_model``, ``nhead``, ``num_layers``,
        ``dropout``, ``T``.

    Returns
    -------
    nn.Module
        The constructed model.
    """
    approach = getattr(cfg, "approach", "pose_transformer")
    num_keypoints = getattr(cfg, "num_keypoints", 543)
    num_classes = getattr(cfg, "num_classes", 100)
    d_model = getattr(cfg, "d_model", 256)
    nhead = getattr(cfg, "nhead", 8)
    num_layers = getattr(cfg, "num_layers", 4)
    dropout = getattr(cfg, "dropout", 0.3)
    T = getattr(cfg, "T", 64)
    use_motion = getattr(cfg, "use_motion", False)

    if approach == "pose_transformer":
        model = PoseTransformer(
            num_keypoints=num_keypoints,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            T=T,
            use_motion=use_motion,
        )
    elif approach == "pose_bilstm":
        model = PoseBiLSTM(
            num_keypoints=num_keypoints,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            T=T,
            use_motion=use_motion,
        )
    else:
        raise ValueError(
            f"Unknown pose approach '{approach}'. "
            "Choose 'pose_transformer' or 'pose_bilstm'."
        )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Built %s with %.2fM parameters", approach, param_count / 1e6)
    return model
