"""
Hybrid fusion model combining pose keypoints and RGB video features.

Supports two fusion strategies:
- ``'concat'``: Late fusion via feature concatenation before a shared FC head.
- ``'attention'``: Cross-attention between pose token embeddings and video
  spatial features, followed by fusion and classification.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """Cross-attention module between pose and video feature sequences.

    Pose tokens attend to video features (and vice versa) using
    multi-head attention, then the attended representations are
    concatenated and projected.

    Parameters
    ----------
    pose_dim : int
        Dimension of pose feature embeddings.
    video_dim : int
        Dimension of video feature embeddings.
    fusion_dim : int
        Output dimension after fusion projection.
    nhead : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        pose_dim: int,
        video_dim: int,
        fusion_dim: int = 256,
        nhead: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Project both modalities to the same dimension for cross-attention
        self.pose_proj = nn.Linear(pose_dim, fusion_dim)
        self.video_proj = nn.Linear(video_dim, fusion_dim)

        # Pose attends to video
        self.pose_to_video_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Video attends to pose
        self.video_to_pose_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_pose = nn.LayerNorm(fusion_dim)
        self.norm_video = nn.LayerNorm(fusion_dim)

        # Final projection: concatenated attended features -> fusion_dim
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, pose_features: torch.Tensor, video_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        pose_features : torch.Tensor
            Shape ``(B, pose_dim)`` -- pooled pose features.
        video_features : torch.Tensor
            Shape ``(B, video_dim)`` -- pooled video features.

        Returns
        -------
        torch.Tensor
            Fused features of shape ``(B, fusion_dim)``.
        """
        # Expand to sequence-of-one for attention compatibility
        pose_q = self.pose_proj(pose_features).unsqueeze(1)  # (B, 1, fusion_dim)
        video_q = self.video_proj(video_features).unsqueeze(1)  # (B, 1, fusion_dim)

        # Cross-attention
        pose_attended, _ = self.pose_to_video_attn(
            query=pose_q, key=video_q, value=video_q
        )
        video_attended, _ = self.video_to_pose_attn(
            query=video_q, key=pose_q, value=pose_q
        )

        pose_attended = self.norm_pose(pose_attended + pose_q)
        video_attended = self.norm_video(video_attended + video_q)

        # Squeeze back and concatenate
        pose_attended = pose_attended.squeeze(1)  # (B, fusion_dim)
        video_attended = video_attended.squeeze(1)  # (B, fusion_dim)

        fused = torch.cat([pose_attended, video_attended], dim=-1)  # (B, 2*fusion_dim)
        return self.output_proj(fused)  # (B, fusion_dim)


class FusionModel(nn.Module):
    """Hybrid fusion model combining pose and video classifiers.

    Uses the ``get_features`` method of each sub-model to extract
    embeddings, then fuses them via concatenation or cross-attention
    before a shared classification head.

    Parameters
    ----------
    pose_model : nn.Module
        Pose-based model (PoseTransformer or PoseBiLSTM) with a
        ``get_features`` method returning ``(B, pose_dim)``.
    video_model : nn.Module
        Video-based model (VideoClassifier) with a ``get_features``
        method returning ``(B, video_dim)``.
    num_classes : int
        Number of output classes.
    fusion : str
        Fusion strategy: ``'concat'`` or ``'attention'``.
    fusion_dim : int
        Intermediate fusion dimension (for attention mode).
    dropout : float
        Dropout rate for the classification head.
    """

    def __init__(
        self,
        pose_model: nn.Module,
        video_model: nn.Module,
        num_classes: int = 100,
        fusion: str = "concat",
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pose_model = pose_model
        self.video_model = video_model
        self.fusion_mode = fusion

        # Determine feature dimensions from sub-models
        self.pose_dim = getattr(pose_model, "d_model", 256)
        self.video_dim = getattr(video_model, "feat_dim", 512)

        if fusion == "concat":
            head_input_dim = self.pose_dim + self.video_dim
            self.fusion_layer = None
        elif fusion == "attention":
            head_input_dim = fusion_dim
            self.fusion_layer = CrossAttentionFusion(
                pose_dim=self.pose_dim,
                video_dim=self.video_dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown fusion mode '{fusion}'. Choose 'concat' or 'attention'."
            )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, num_classes),
        )

    def forward(
        self, pose_input: torch.Tensor, video_input: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        pose_input : torch.Tensor
            Keypoint input of shape ``(B, T, num_keypoints * 3)``.
        video_input : torch.Tensor
            Video input of shape ``(B, C, T, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        pose_features = self.pose_model.get_features(pose_input)  # (B, pose_dim)
        video_features = self.video_model.get_features(video_input)  # (B, video_dim)

        if self.fusion_mode == "concat":
            fused = torch.cat([pose_features, video_features], dim=-1)
        else:
            fused = self.fusion_layer(pose_features, video_features)

        logits = self.head(fused)
        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_fusion_model(
    cfg: Any, pose_model: nn.Module, video_model: nn.Module
) -> nn.Module:
    """Build a fusion model from a configuration and pre-built sub-models.

    Parameters
    ----------
    cfg : object
        Configuration with attributes: ``num_classes``, ``fusion``,
        ``fusion_dim``, ``dropout``.
    pose_model : nn.Module
        Pre-built pose model.
    video_model : nn.Module
        Pre-built video model.

    Returns
    -------
    FusionModel
    """
    num_classes = getattr(cfg, "num_classes", 100)
    fusion = getattr(cfg, "fusion", "concat")
    fusion_dim = getattr(cfg, "fusion_dim", 256)
    dropout = getattr(cfg, "dropout", 0.3)

    model = FusionModel(
        pose_model=pose_model,
        video_model=video_model,
        num_classes=num_classes,
        fusion=fusion,
        fusion_dim=fusion_dim,
        dropout=dropout,
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built FusionModel(%s) with %.2fM parameters", fusion, param_count / 1e6
    )
    return model
