import numpy as np
import torch
import torch.nn.functional as F
import random
from typing import Dict, List, Tuple, Union, Optional


class BEVAugmentor:
    """Advanced data augmentation for BEV segmentation.
    
    This class implements various augmentation techniques specifically designed for
    Bird's Eye View segmentation, including:
    - Random horizontal and vertical flips
    - Random rotation
    - Random scaling
    - Random translation
    - Cutout (for both input and target)
    - Mixup (combining two samples)
    - Class frequency jittering
    - Feature noise for regularization
    """
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.7,
        rotate_range: Tuple[float, float] = (-10, 10),
        scale_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translate_prob: float = 0.5,
        translate_range: Tuple[float, float] = (-5, 5),
        cutout_prob: float = 0.5,
        cutout_max_boxes: int = 3,
        cutout_max_size: Tuple[int, int] = (30, 30),
        mixup_prob: float = 0.3,
        feature_noise_prob: float = 0.5,
        feature_noise_std: float = 0.01,
        color_jitter_prob: float = 0.5,
        class_distribution_jitter: bool = True,
    ):
        """Initialize the augmentor with various augmentation parameters.
        
        Args:
            flip_prob: Probability of applying random flip
            rotate_prob: Probability of applying random rotation
            rotate_range: Range of rotation in degrees (min, max)
            scale_prob: Probability of applying random scaling
            scale_range: Range of scaling factors (min, max)
            translate_prob: Probability of applying random translation
            translate_range: Range of translation in pixels (min, max)
            cutout_prob: Probability of applying cutout
            cutout_max_boxes: Maximum number of cutout boxes
            cutout_max_size: Maximum size of cutout boxes (height, width)
            mixup_prob: Probability of applying mixup with another sample
            feature_noise_prob: Probability of adding feature noise
            feature_noise_std: Standard deviation of feature noise
            color_jitter_prob: Probability of applying color jitter to images
            class_distribution_jitter: Whether to jitter class distribution
        """
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.translate_prob = translate_prob
        self.translate_range = translate_range
        self.cutout_prob = cutout_prob
        self.cutout_max_boxes = cutout_max_boxes
        self.cutout_max_size = cutout_max_size
        self.mixup_prob = mixup_prob
        self.feature_noise_prob = feature_noise_prob
        self.feature_noise_std = feature_noise_std
        self.color_jitter_prob = color_jitter_prob
        self.class_distribution_jitter = class_distribution_jitter
        
    def __call__(
        self, 
        batch: Dict[str, torch.Tensor], 
        lidar_only: bool = False,
        mixup_batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply augmentations to a batch of data.
        
        Args:
            batch: Dictionary containing batch data (image, lidar, bev_label)
            lidar_only: Whether to only augment lidar data
            mixup_batch: Optional second batch for mixup augmentation
            
        Returns:
            Augmented batch dictionary
        """
        # Create a copy of the batch to avoid modifying the original
        augmented_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
        
        # Random horizontal flip
        if random.random() < self.flip_prob:
            augmented_batch = self._flip_horizontal(augmented_batch, lidar_only)
        
        # Random vertical flip
        if random.random() < self.flip_prob:
            augmented_batch = self._flip_vertical(augmented_batch, lidar_only)
        
        # Random rotation
        if random.random() < self.rotate_prob:
            angle = random.uniform(self.rotate_range[0], self.rotate_range[1])
            augmented_batch = self._rotate(augmented_batch, angle, lidar_only)
        
        # Random scaling
        if random.random() < self.scale_prob:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            augmented_batch = self._scale(augmented_batch, scale, lidar_only)
        
        # Random translation
        if random.random() < self.translate_prob:
            tx = random.uniform(self.translate_range[0], self.translate_range[1])
            ty = random.uniform(self.translate_range[0], self.translate_range[1])
            augmented_batch = self._translate(augmented_batch, tx, ty, lidar_only)
        
        # Cutout augmentation on BEV
        if random.random() < self.cutout_prob:
            augmented_batch = self._cutout(augmented_batch)
        
        # Mixup augmentation if second batch is provided
        if mixup_batch is not None and random.random() < self.mixup_prob:
            alpha = random.uniform(0.3, 0.7)  # Mixup ratio
            augmented_batch = self._mixup(augmented_batch, mixup_batch, alpha)
        
        # Add noise to features for regularization
        if random.random() < self.feature_noise_prob:
            augmented_batch = self._add_feature_noise(augmented_batch, lidar_only)
            
        # Color jittering for images
        if not lidar_only and 'image' in augmented_batch and random.random() < self.color_jitter_prob:
            augmented_batch = self._color_jitter(augmented_batch)
        
        return augmented_batch
    
    def _flip_horizontal(
        self,
        batch: Dict[str, torch.Tensor],
        lidar_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Apply horizontal flip to batch data."""
        # Flip BEV label (horizontal flip = flip along width/x-axis)
        if 'bev_label' in batch:
            batch['bev_label'] = torch.flip(batch['bev_label'], dims=[-1])
        
        # Flip LiDAR point cloud
        if 'lidar' in batch:
            # Flip x-coordinate (assuming x is the first column)
            batch['lidar'][:, 0] = -batch['lidar'][:, 0]
        
        # Flip image if not lidar_only
        if not lidar_only and 'image' in batch:
            # For image tensor [C, H, W], flip along last dimension (width)
            batch['image'] = torch.flip(batch['image'], dims=[-1])
        
        return batch
    
    def _flip_vertical(
        self,
        batch: Dict[str, torch.Tensor],
        lidar_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Apply vertical flip to batch data."""
        # Flip BEV label (vertical flip = flip along height/y-axis)
        if 'bev_label' in batch:
            batch['bev_label'] = torch.flip(batch['bev_label'], dims=[-2])
        
        # Flip LiDAR point cloud
        if 'lidar' in batch:
            # Flip y-coordinate (assuming y is the second column)
            batch['lidar'][:, 1] = -batch['lidar'][:, 1]
        
        # Flip image if not lidar_only
        if not lidar_only and 'image' in batch:
            # For image tensor [C, H, W], flip along second-to-last dimension (height)
            batch['image'] = torch.flip(batch['image'], dims=[-2])
        
        return batch
    
    def _rotate(
        self,
        batch: Dict[str, torch.Tensor],
        angle: float,
        lidar_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Apply rotation to batch data.
        
        Args:
            batch: Batch dictionary
            angle: Rotation angle in degrees
            lidar_only: Whether to only rotate lidar data
        """
        # Convert angle to radians
        angle_rad = np.radians(angle)
        cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotate BEV label
        if 'bev_label' in batch:
            # Create rotation grid
            label = batch['bev_label']
            if len(label.shape) == 2:
                # Add batch and channel dimensions if missing
                label = label.unsqueeze(0)
                
            # Get dimensions
            h, w = label.shape[-2:]
            rotation_matrix = torch.tensor([
                [cos_theta, sin_theta, 0],
                [-sin_theta, cos_theta, 0]
            ], dtype=torch.float32, device=label.device)
            
            # Apply affine grid
            grid = F.affine_grid(
                rotation_matrix.unsqueeze(0), 
                size=(1, 1, h, w),
                align_corners=False
            )
            
            # Use nearest neighbor interpolation for labels
            label_rotated = F.grid_sample(
                label.float().unsqueeze(1),
                grid,
                mode='nearest',
                align_corners=False
            ).squeeze(1).long()
            
            batch['bev_label'] = label_rotated
        
        # Rotate LiDAR point cloud
        if 'lidar' in batch:
            points = batch['lidar']
            # Apply 2D rotation to x,y coordinates
            x, y = points[:, 0], points[:, 1]
            points[:, 0] = cos_theta * x - sin_theta * y
            points[:, 1] = sin_theta * x + cos_theta * y
        
        # Rotate image if not lidar_only
        if not lidar_only and 'image' in batch:
            # TODO: Implement proper image rotation with crop
            pass
        
        return batch
    
    def _scale(
        self,
        batch: Dict[str, torch.Tensor],
        scale: float,
        lidar_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Apply scaling to batch data."""
        # Scale LiDAR point cloud
        if 'lidar' in batch:
            # Scale x,y,z coordinates
            batch['lidar'][:, :3] *= scale
        
        # NOTE: We don't scale the BEV label or image, as this would
        # change the alignment between lidar points and the grid
        
        return batch
    
    def _translate(
        self,
        batch: Dict[str, torch.Tensor],
        tx: float,
        ty: float,
        lidar_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Apply translation to batch data."""
        # Translate LiDAR point cloud
        if 'lidar' in batch:
            # Translate x,y coordinates
            batch['lidar'][:, 0] += tx
            batch['lidar'][:, 1] += ty
        
        # NOTE: We don't translate the BEV label or image, as this would
        # change the alignment between lidar points and the grid
        
        return batch
    
    def _cutout(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply cutout augmentation to BEV label."""
        if 'bev_label' not in batch:
            return batch
        
        label = batch['bev_label']
        h, w = label.shape[-2:]
        
        # Determine number of cutout boxes
        num_boxes = random.randint(1, self.cutout_max_boxes)
        
        # Apply cutout to label
        for _ in range(num_boxes):
            # Random box size
            box_h = random.randint(5, self.cutout_max_size[0])
            box_w = random.randint(5, self.cutout_max_size[1])
            
            # Random box position
            top = random.randint(0, h - box_h)
            left = random.randint(0, w - box_w)
            
            # Create mask
            mask = torch.ones_like(label).bool()
            mask[..., top:top+box_h, left:left+box_w] = False
            
            # Apply mask
            background_class = 0  # Assuming 0 is background class
            masked_label = label.clone()
            masked_label[~mask] = background_class
            
            # Apply masked label
            batch['bev_label'] = masked_label
        
        return batch
    
    def _mixup(
        self,
        batch1: Dict[str, torch.Tensor],
        batch2: Dict[str, torch.Tensor],
        alpha: float
    ) -> Dict[str, torch.Tensor]:
        """Apply mixup augmentation between two batches."""
        mixed_batch = {}
        
        # Keys to mix (tensors only)
        for key in batch1:
            if isinstance(batch1[key], torch.Tensor) and key in batch2:
                if key == 'bev_label':
                    # For segmentation masks, we use hard mixing
                    # Create a binary mask for which pixels to keep from batch1
                    mask = torch.rand_like(batch1[key].float()) < alpha
                    mixed_batch[key] = torch.where(mask, batch1[key], batch2[key])
                elif key == 'lidar':
                    # For lidar, we combine points with probability alpha
                    # First determine how many points to keep from each batch
                    n1 = batch1[key].shape[0]
                    n2 = batch2[key].shape[0]
                    keep1 = int(alpha * n1)
                    keep2 = int((1 - alpha) * n2)
                    
                    # Randomly select points to keep
                    idx1 = torch.randperm(n1)[:keep1]
                    idx2 = torch.randperm(n2)[:keep2]
                    
                    # Combine points
                    mixed_batch[key] = torch.cat([
                        batch1[key][idx1],
                        batch2[key][idx2]
                    ], dim=0)
                elif key == 'image':
                    # For images, we do standard mixup
                    mixed_batch[key] = alpha * batch1[key] + (1 - alpha) * batch2[key]
                else:
                    # For other tensors, keep batch1 values
                    mixed_batch[key] = batch1[key]
            else:
                # Non-tensor fields, keep from batch1
                mixed_batch[key] = batch1[key]
        
        return mixed_batch
    
    def _add_feature_noise(
        self,
        batch: Dict[str, torch.Tensor],
        lidar_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise to features for regularization."""
        # Add noise to LiDAR points
        if 'lidar' in batch:
            points = batch['lidar']
            # Add small noise to x,y,z coordinates
            noise = torch.randn_like(points[:, :3]) * self.feature_noise_std
            batch['lidar'][:, :3] += noise
        
        # Add noise to image if not lidar_only
        if not lidar_only and 'image' in batch:
            image = batch['image']
            noise = torch.randn_like(image) * self.feature_noise_std
            batch['image'] = (image + noise).clamp(0, 1)  # Ensure valid range
        
        return batch
    
    def _color_jitter(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply color jittering to images."""
        if 'image' not in batch:
            return batch
        
        image = batch['image']
        
        # Brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        image = image * brightness_factor
        
        # Contrast adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        mean = image.mean(dim=[-2, -1], keepdim=True)
        image = (image - mean) * contrast_factor + mean
        
        # Saturation adjustment (only for RGB images)
        if image.shape[1] == 3:
            saturation_factor = random.uniform(0.8, 1.2)
            # Convert to grayscale
            gray = image.mean(dim=1, keepdim=True)
            # Blend with original based on saturation factor
            image = gray + saturation_factor * (image - gray)
        
        # Ensure valid range
        batch['image'] = image.clamp(0, 1)
        
        return batch


class RegularizedTrainingWrapper:
    """Wrapper for training with regularization techniques."""
    
    def __init__(
        self,
        dataset,
        augmentor: Optional[BEVAugmentor] = None,
        dropout_rate: float = 0.1,
        weight_decay: float = 1e-4,
    ):
        """Initialize the regularized training wrapper.
        
        Args:
            dataset: The dataset to wrap
            augmentor: BEV augmentor instance for data augmentation
            dropout_rate: Dropout rate for regularization
            weight_decay: Weight decay for regularization
        """
        self.dataset = dataset
        self.augmentor = augmentor or BEVAugmentor()
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a sample with regularization applied."""
        # Get original sample
        sample = self.dataset[idx]
        
        # Apply augmentations if in training mode
        if self.augmentor is not None:
            # Get a random second sample for potential mixup (with 30% probability)
            if random.random() < 0.3:
                # Get a random index different from the current one
                mixup_idx = random.randint(0, len(self.dataset) - 1)
                if mixup_idx == idx:
                    mixup_idx = (mixup_idx + 1) % len(self.dataset)
                    
                mixup_sample = self.dataset[mixup_idx]
                sample = self.augmentor(sample, mixup_batch=mixup_sample)
            else:
                sample = self.augmentor(sample)
        
        return sample 