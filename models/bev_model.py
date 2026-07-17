"""End-to-end BEV fusion model.

This module wraps the camera backbone, LiDAR backbone, per-stage projection
layers, cross-attention fusion, and segmentation head into a single
``nn.Module`` that mirrors the forward path used during training in
``train.py``. Both ``test.py`` and ``inference.py`` build on this so that the
evaluation / inference path can never silently drift from the trained model.

The state dict layout matches the checkpoints saved by ``train.py``:
``camera_backbone``, ``lidar_backbone``, ``fusion``, ``head``,
``image_projections``, and ``lidar_projections``.
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from models.backbones import EfficientNetV2Backbone, SECONDBackbone
from models.fusion import BEVFusion
from models.heads import BEVSegmentationHead


# Channel dimensions at each fusion stage. These are the projected channels the
# fusion module expects (image queries: 32/64/128, LiDAR keys/values:
# 64/128/256) and must match the layers below and the trained checkpoint.
STAGE_CHANNELS = {
    'stage1': {'lidar': 64, 'image': 32},    # camera stage3 -> fusion stage1
    'stage2': {'lidar': 128, 'image': 64},   # camera stage4 -> fusion stage2
    'stage3': {'lidar': 256, 'image': 128},  # camera stage5 -> fusion stage3
}

# Maps the camera backbone's stage names to the fusion module's stage names.
CONFIG_TO_FUSION_STAGE = {
    'stage3': 'stage1',
    'stage4': 'stage2',
    'stage5': 'stage3',
}


def _projection(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class BEVFusionModel(nn.Module):
    """Single-camera + single-LiDAR BEV semantic segmentation model."""

    def __init__(
        self,
        num_classes: int = 6,
        voxel_size=(0.8, 0.8, 0.8),
        point_cloud_range=(-51.2, -51.2, -5, 51.2, 51.2, 3),
        max_num_points: int = 32,
        max_voxels: int = 20000,
        pretrained_camera: bool = False,
        hidden_channels: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.camera_backbone = EfficientNetV2Backbone(pretrained=pretrained_camera)

        self.lidar_backbone = SECONDBackbone(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

        # Project raw backbone features to the channel dims the fusion expects.
        self.image_projections = nn.ModuleDict({
            'stage1': _projection(128, 32),
            'stage2': _projection(256, 64),
            'stage3': _projection(512, 128),
        })
        self.lidar_projections = nn.ModuleDict({
            'stage1': _projection(64, 64),
            'stage2': _projection(128, 128),
            'stage3': _projection(256, 256),
        })

        self.fusion = BEVFusion(
            lidar_channels=128,
            output_channels=128,
            spatial_size=(128, 128),
            stage_channels=STAGE_CHANNELS,
        )

        self.head = BEVSegmentationHead(
            in_channels=128,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def _extract_lidar_features(
        self, points_list: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Run the LiDAR backbone over a list of point clouds and stack BEV features."""
        feats = []
        for points in points_list:
            if points.dim() == 2:
                points = points.unsqueeze(0)
            feats.append(self.lidar_backbone(points)['bev_features'])

        stages = ['stage1', 'stage2', 'stage3']
        same_shape = all(
            f['stage3'].shape == feats[0]['stage3'].shape for f in feats
        )
        if same_shape:
            return {s: torch.cat([f[s] for f in feats], dim=0) for s in stages}
        # Fallback: shapes differ (should not happen with a fixed BEV grid).
        return {s: feats[0][s] for s in stages}

    def forward(
        self,
        images: torch.Tensor,
        points_list: List[torch.Tensor],
        targets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            images: [B, 3, H, W] camera images.
            points_list: list of B point-cloud tensors ([N_i, 5]).
            targets: optional [B, H, W] class-index labels. If given, returns the
                loss dict from the head; otherwise returns per-class probabilities.
        Returns:
            Loss dict (if ``targets`` given) or [B, num_classes, H, W] probabilities.
        """
        image_feats = self.camera_backbone(images)
        lidar_feats = self._extract_lidar_features(points_list)

        lidar_features_dict = {
            stage: self.lidar_projections[stage](lidar_feats[stage])
            for stage in ('stage1', 'stage2', 'stage3')
        }

        image_features_dict = {}
        for config_stage, fusion_stage in CONFIG_TO_FUSION_STAGE.items():
            src = config_stage if config_stage in image_feats else 'stage5'
            image_features_dict[fusion_stage] = self.image_projections[fusion_stage](
                image_feats[src]
            )

        fused_feats, _ = self.fusion(lidar_features_dict, image_features_dict)
        return self.head(fused_feats, targets)

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, points_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Return a [B, H, W] tensor of predicted class indices."""
        probs = self.forward(images, points_list)
        return probs.argmax(dim=1)

    def load_checkpoint(self, path: str, map_location='cpu') -> dict:
        """Load a checkpoint saved by ``train.py`` (component-wise state dicts)."""
        checkpoint = torch.load(path, map_location=map_location)
        self.camera_backbone.load_state_dict(checkpoint['camera_backbone'])
        self.lidar_backbone.load_state_dict(checkpoint['lidar_backbone'])
        self.fusion.load_state_dict(checkpoint['fusion'])
        self.head.load_state_dict(checkpoint['head'])
        if 'image_projections' in checkpoint:
            self.image_projections.load_state_dict(checkpoint['image_projections'])
        if 'lidar_projections' in checkpoint:
            self.lidar_projections.load_state_dict(checkpoint['lidar_projections'])
        return checkpoint

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        model_config: Optional[dict] = None,
        device: Union[str, torch.device] = 'cpu',
    ) -> 'BEVFusionModel':
        """Build a model from a config dict and load weights from ``path``."""
        model_config = model_config or {}
        seg = model_config.get('segmentation_head', {})
        lidar = model_config.get('lidar_backbone', {})

        model = cls(
            num_classes=seg.get('num_classes', 6),
            voxel_size=lidar.get('voxel_size', (0.8, 0.8, 0.8)),
            point_cloud_range=lidar.get(
                'point_cloud_range', (-51.2, -51.2, -5, 51.2, 51.2, 3)
            ),
            max_num_points=lidar.get('max_num_points', 32),
            max_voxels=lidar.get('max_voxels', 20000),
            pretrained_camera=False,
            dropout=0.0,
        )
        model.load_checkpoint(path, map_location=device)
        model.to(device)
        model.eval()
        return model
