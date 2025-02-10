import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2


class ImageTransform:
    """
    Image preprocessing for ZED camera input
    Supports:
    - Resizing
    - Normalization
    - Color augmentation
    - Geometric augmentation
    """
    def __init__(
        self,
        size: Tuple[int, int],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        training: bool = True
    ):
        self.size = size
        self.mean = mean
        self.std = std
        self.training = training
        
        # Basic transforms
        self.normalize = T.Normalize(mean=mean, std=std)
        
        # Training augmentations
        if training:
            self.color_jitter = T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Process image
        Args:
            image: [C, H, W] tensor in range [0, 1]
        Returns:
            Processed image tensor
        """
        # Resize if needed
        if image.shape[-2:] != self.size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Apply color augmentation in training
        if self.training:
            image = self.color_jitter(image)
        
        # Normalize
        image = self.normalize(image)
        
        return image


class PointCloudTransform:
    """
    Point cloud preprocessing for Velodyne LiDAR
    Supports:
    - Voxelization
    - Random sampling
    - Geometric augmentation
    """
    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        max_points_per_voxel: int = 32,
        max_voxels: int = 4000,
        training: bool = True
    ):
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_points = max_points_per_voxel
        self.max_voxels = max_voxels
        self.training = training
        
        # Calculate grid size
        self.grid_size = np.round(
            (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        ).astype(np.int64)
    
    def __call__(
        self,
        points: torch.Tensor,
        augment: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process point cloud
        Args:
            points: [N, 4] tensor of (x, y, z, intensity)
            augment: Whether to apply augmentation
        Returns:
            Dictionary containing:
            - voxels: Voxelized points
            - coordinates: Voxel coordinates
            - num_points: Number of points per voxel
        """
        if augment and self.training:
            points = self.augment_points(points)
        
        # Filter points outside range
        mask = self.filter_points(points)
        points = points[mask]
        
        # Voxelize points
        voxels, coords, num_points = self.voxelize(points)
        
        return {
            'voxels': voxels,
            'coordinates': coords,
            'num_points': num_points
        }
    
    def filter_points(self, points: torch.Tensor) -> torch.Tensor:
        """Filter points within range"""
        point_range = self.point_cloud_range
        mask = (
            (points[:, 0] >= point_range[0]) &
            (points[:, 0] <= point_range[3]) &
            (points[:, 1] >= point_range[1]) &
            (points[:, 1] <= point_range[4]) &
            (points[:, 2] >= point_range[2]) &
            (points[:, 2] <= point_range[5])
        )
        return mask
    
    def voxelize(
        self,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert points to voxels
        Args:
            points: [N, 4] tensor of points
        Returns:
            voxels: [M, max_points, 4] tensor of voxelized points
            coords: [M, 4] tensor of voxel coordinates (batch, z, y, x)
            num_points: [M] tensor of number of points per voxel
        """
        # Convert points to voxel coordinates
        voxel_coords = (
            (points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
        ).int()
        
        # Get unique voxels
        voxel_coords, inverse_indices, voxel_counts = torch.unique(
            voxel_coords, dim=0, return_inverse=True, return_counts=True
        )
        
        # Limit number of voxels
        if len(voxel_coords) > self.max_voxels:
            indices = torch.randperm(len(voxel_coords))[:self.max_voxels]
            voxel_coords = voxel_coords[indices]
            voxel_counts = voxel_counts[indices]
            mask = torch.isin(inverse_indices, indices)
            inverse_indices = torch.searchsorted(indices, inverse_indices[mask])
            points = points[mask]
        
        # Initialize voxel tensors
        voxels = torch.zeros(
            len(voxel_coords), self.max_points, 4,
            dtype=points.dtype, device=points.device
        )
        num_points = torch.zeros(
            len(voxel_coords),
            dtype=torch.int32, device=points.device
        )
        
        # Fill voxels
        for i, count in enumerate(voxel_counts):
            voxel_mask = inverse_indices == i
            voxel_points = points[voxel_mask]
            
            # Limit points per voxel
            if count > self.max_points:
                indices = torch.randperm(count)[:self.max_points]
                voxel_points = voxel_points[indices]
                count = self.max_points
            
            voxels[i, :count] = voxel_points
            num_points[i] = count
        
        return voxels, voxel_coords, num_points
    
    def augment_points(self, points: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to points"""
        if not self.training:
            return points
        
        # Random rotation around z-axis
        if torch.rand(1) > 0.5:
            angle = torch.rand(1) * 2 * np.pi
            rot_sin = torch.sin(angle)
            rot_cos = torch.cos(angle)
            
            x = points[:, 0]
            y = points[:, 1]
            points[:, 0] = rot_cos * x - rot_sin * y
            points[:, 1] = rot_sin * x + rot_cos * y
        
        # Random scaling
        if torch.rand(1) > 0.5:
            scale = 1 + (torch.rand(1) * 0.2 - 0.1)  # Scale by ±10%
            points[:, :3] *= scale
        
        # Random translation
        if torch.rand(1) > 0.5:
            translation = (torch.rand(3) * 0.2 - 0.1)  # Translate by ±0.1m
            points[:, :3] += translation
        
        return points


class LabelTransform:
    """
    Label preprocessing for BEV segmentation
    Supports:
    - Label rasterization
    - One-hot encoding
    - Label smoothing
    """
    def __init__(
        self,
        num_classes: int,
        grid_size: Tuple[int, int],
        label_smoothing: float = 0.1
    ):
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.label_smoothing = label_smoothing
    
    def __call__(
        self,
        labels: torch.Tensor,
        smooth: bool = True
    ) -> torch.Tensor:
        """
        Process segmentation labels
        Args:
            labels: [H, W] tensor of class indices
            smooth: Whether to apply label smoothing
        Returns:
            [num_classes, H, W] one-hot encoded labels
        """
        # Resize if needed
        if labels.shape != self.grid_size:
            labels = F.interpolate(
                labels.unsqueeze(0).unsqueeze(0).float(),
                size=self.grid_size,
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
        
        # One-hot encode
        one_hot = F.one_hot(labels, num_classes=self.num_classes)
        one_hot = one_hot.permute(2, 0, 1).float()
        
        # Apply label smoothing
        if smooth and self.label_smoothing > 0:
            one_hot = one_hot * (1 - self.label_smoothing) + \
                     self.label_smoothing / self.num_classes
        
        return one_hot 