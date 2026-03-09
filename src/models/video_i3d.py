"""
RGB video classifier using pretrained 3D CNN backbones.

Supports backbones from both ``torchvision.models.video`` (r2plus1d_18,
r3d_18, mc3_18) and ``pytorchvideo`` (slow_r50, slowfast_r50, x3d_m).
The final classification head is replaced with a new linear layer sized
for the target number of classes.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Mapping of backbone names to their feature dimensions
_TORCHVISION_BACKBONES = {
    "r2plus1d_18": 512,
    "r3d_18": 512,
    "mc3_18": 512,
}

_PYTORCHVIDEO_BACKBONES = {
    "slow_r50": 2048,
    "slowfast_r50": 2304,
    "x3d_m": 192,   # X3D-M backbone output dim before the projection head
}


class VideoClassifier(nn.Module):
    """Video classification model using pretrained 3D backbones.

    The backbone's original classification head is removed and replaced
    with a dropout layer and a new linear head for the target number of
    classes.

    Parameters
    ----------
    backbone : str
        Name of the backbone.  Supported:
        ``'r2plus1d_18'``, ``'r3d_18'``, ``'mc3_18'`` (torchvision),
        ``'slow_r50'``, ``'slowfast_r50'``, ``'x3d_m'`` (pytorchvideo).
    num_classes : int
        Number of output classes.
    pretrained : bool
        Whether to load pretrained weights.
    dropout : float
        Dropout rate before the classification head.
    """

    def __init__(
        self,
        backbone: str = "r2plus1d_18",
        num_classes: int = 100,
        pretrained: bool = True,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone in _TORCHVISION_BACKBONES:
            self.feat_dim = _TORCHVISION_BACKBONES[backbone]
            self.backbone, self.is_slowfast = self._build_torchvision(backbone, pretrained)
        elif backbone in _PYTORCHVIDEO_BACKBONES:
            self.feat_dim = _PYTORCHVIDEO_BACKBONES[backbone]
            self.backbone, self.is_slowfast = self._build_pytorchvideo(backbone, pretrained)
        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Supported: "
                f"{list(_TORCHVISION_BACKBONES) + list(_PYTORCHVIDEO_BACKBONES)}"
            )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes),
        )

    def _build_torchvision(
        self, backbone: str, pretrained: bool
    ) -> tuple[nn.Module, bool]:
        """Build a torchvision video backbone with the head removed.

        Returns
        -------
        tuple[nn.Module, bool]
            The backbone module and whether it is a SlowFast model.
        """
        import torchvision.models.video as video_models

        weights_arg = "DEFAULT" if pretrained else None

        if backbone == "r2plus1d_18":
            model = video_models.r2plus1d_18(weights=weights_arg)
        elif backbone == "r3d_18":
            model = video_models.r3d_18(weights=weights_arg)
        elif backbone == "mc3_18":
            model = video_models.mc3_18(weights=weights_arg)
        else:
            raise ValueError(f"Unsupported torchvision backbone: {backbone}")

        # Remove the original FC layer; keep everything up to avgpool
        model.fc = nn.Identity()
        return model, False

    def _build_pytorchvideo(
        self, backbone: str, pretrained: bool
    ) -> tuple[nn.Module, bool]:
        """Build a pytorchvideo backbone with the head removed.

        Returns
        -------
        tuple[nn.Module, bool]
            The backbone module and whether it is a SlowFast model.
        """
        try:
            import pytorchvideo.models.hub as pvhub  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"pytorchvideo is required for backbone '{backbone}'. "
                "Install it with: pip install pytorchvideo"
            ) from exc

        is_slowfast = backbone == "slowfast_r50"

        if backbone == "slow_r50":
            model = torch.hub.load(
                "facebookresearch/pytorchvideo",
                "slow_r50",
                pretrained=pretrained,
            )
        elif backbone == "slowfast_r50":
            model = torch.hub.load(
                "facebookresearch/pytorchvideo",
                "slowfast_r50",
                pretrained=pretrained,
            )
        elif backbone == "x3d_m":
            model = torch.hub.load(
                "facebookresearch/pytorchvideo",
                "x3d_m",
                pretrained=pretrained,
            )
        else:
            raise ValueError(f"Unsupported pytorchvideo backbone: {backbone}")

        # PyTorchVideo models end with a head block (ResNetBasicHead / X3DHead)
        # that contains pooling, projection, and activation.  Replace the entire
        # head block with AdaptiveAvgPool3d + Flatten to get a reliable (B, feat_dim)
        # output whose dimension matches _PYTORCHVIDEO_BACKBONES.
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            model.blocks[-1] = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(1),
            )

        return model, is_slowfast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input video tensor of shape ``(B, C, T, H, W)``.
            For SlowFast models, this is automatically split into slow
            and fast pathways.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        if self.is_slowfast:
            # SlowFast expects a list: [slow_pathway, fast_pathway]
            # slow: sample every 4th frame; fast: all frames
            slow = x[:, :, ::4, :, :]
            fast = x
            features = self.backbone([slow, fast])
        else:
            features = self.backbone(x)

        # Ensure features are 2D (B, feat_dim)
        if features.dim() > 2:
            features = features.flatten(1)

        logits = self.head(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before the classification head.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, C, T, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, feat_dim)``.
        """
        if self.is_slowfast:
            slow = x[:, :, ::4, :, :]
            fast = x
            features = self.backbone([slow, fast])
        else:
            features = self.backbone(x)

        if features.dim() > 2:
            features = features.flatten(1)
        return features


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_video_model(cfg: Any) -> nn.Module:
    """Build a video classifier from a configuration object.

    Parameters
    ----------
    cfg : object
        Configuration with attributes: ``backbone``, ``num_classes``,
        ``pretrained``, ``dropout``.

    Returns
    -------
    VideoClassifier
    """
    backbone = getattr(cfg, "backbone", "r2plus1d_18")
    num_classes = getattr(cfg, "num_classes", 100)
    pretrained = getattr(cfg, "pretrained", True)
    dropout = getattr(cfg, "dropout", 0.4)

    model = VideoClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built VideoClassifier(%s) with %.2fM parameters",
        backbone,
        param_count / 1e6,
    )
    return model
