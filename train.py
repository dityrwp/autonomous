# train.py

import os
import time
import argparse
from typing import Dict, Optional, List, Tuple
import datetime
import shutil
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
import numpy as np
from tqdm import tqdm
import cv2
import math
import logging
import matplotlib.pyplot as plt # Added for potential visualizations
import random

# Assume these imports are correct based on your project structure
from models.backbones import EfficientNetV2Backbone, SECONDBackbone
from models.fusion import BEVFusion, DepthAugmentedBEVLifter
from models.heads import BEVSegmentationHead
from datasets.precomputed_bev_dataset import PrecomputedBEVDataset
from datasets.augmentation import BEVAugmentor, RegularizedTrainingWrapper
from utils.metrics import SegmentationMetrics, MetricsLogger
from utils.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add this helper function near the top or before the Trainer class
def move_to_device(data, device):
    """Recursively move tensors in a nested dictionary or list to the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(x, device) for x in data]
    else:
        return data

class Trainer:
    def __init__(
        self,
        config: dict, # Combined model+train config passed from main()
        train_config: dict, # Specific training config part
        model_config: dict, # Specific model config part
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = None,
        output_dir: Path = None,
        use_early_stopping: bool = True,
        early_stopping_patience: int = 5,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize trainer."""
        # Store configuration
        self.config = config # Combined config
        self.train_config = train_config
        self.model_config = model_config # Store model config separately
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir) if output_dir else Path('outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup checkpoint directory
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Setup visualization directory
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Training setup
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0

        # Get optimizer config
        self.opt_config = train_config.get('optimizer', {})
        self.scheduler_config = train_config.get('scheduler', {})

        # Setting learning rates and regularization
        self.lr_camera = self.opt_config.get('lr_camera', 5e-5)
        self.lr_lidar = self.opt_config.get('lr_lidar', 5e-5)
        self.lr_fusion = self.opt_config.get('lr_fusion', 5e-5)
        self.lr_head = self.opt_config.get('lr_head', 5e-5)
        self.weight_decay = weight_decay or self.opt_config.get('weight_decay', 1e-4)
        self.dropout_rate = dropout_rate

        # Pass dropout rate to head config
        if 'segmentation_head' in self.model_config:
            self.model_config['segmentation_head']['dropout'] = self.dropout_rate
            # Also update the combined config for consistency if needed by head directly
            self.config['segmentation_head'] = self.model_config['segmentation_head']


        # Set up model and optimizer
        # _create_model now returns a dict of components
        self.model_components = self._create_model()
        self.camera_backbone = self.model_components['camera_backbone']
        self.lidar_backbone = self.model_components['lidar_backbone']
        self.image_lifter = self.model_components['image_lifter']
        self.final_lidar_projection = self.model_components['final_lidar_projection']
        self.fusion = self.model_components['fusion']
        self.head = self.model_components['head']
        # self.model refers to the primary component for checkpointing/early stopping (e.g., the head)
        self.model = self.head
        print(f"Initialized model components on {self.device}")

        # Move all components to device
        for component in self.model_components.values():
            if isinstance(component, nn.Module):
                component.to(self.device)

        # Print model parameters
        self._log_model_info() # Log component-wise parameters

        # --- Metrics ---
        num_classes = self.model_config.get('segmentation_head', {}).get('num_classes', 6)
        self.train_metrics = SegmentationMetrics(num_classes=num_classes, device=self.device)
        self.val_metrics = SegmentationMetrics(num_classes=num_classes, device=self.device)
        self.metrics_logger = MetricsLogger() # Assuming this class exists and works

        # --- Optimizer ---
        # Include new single projection layers in optimizer parameters
        self.optimizer = torch.optim.AdamW([
            {'params': self.camera_backbone.parameters(), 'lr': self.lr_camera},
            {'params': self.lidar_backbone.parameters(), 'lr': self.lr_lidar},
            {'params': self.image_lifter.parameters(), 'lr': self.lr_fusion},  # BEV lifter uses fusion LR
            {'params': self.final_lidar_projection.parameters(), 'lr': self.lr_fusion},  # Final projection uses fusion LR
            {'params': self.fusion.parameters(), 'lr': self.lr_fusion},
            {'params': self.head.parameters(), 'lr': self.lr_head},
        ], weight_decay=self.weight_decay)

        # --- Scheduler ---
        scheduler_name = self.scheduler_config.get('name', 'CosineAnnealingLR')
        epochs = self.train_config.get('train', {}).get('epochs', 50)
        if scheduler_name == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=self.scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Monitor validation IoU
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 3),
                min_lr=self.scheduler_config.get('eta_min', 1e-6)
            )
        else:
            print(f"Warning: Scheduler '{scheduler_name}' not recognized or 'None'. No scheduler used.")
            self.scheduler = None

        # --- AMP Scaler ---
        self.scaler = amp.GradScaler(enabled=self.train_config.get('train', {}).get('mixed_precision', True))

        # --- State Tracking ---
        self.best_val_iou = 0.0
        self.train_step = 0
        self.epoch = 0

        # --- Early Stopping ---
        self.use_early_stopping = use_early_stopping
        if use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                mode='max', # Monitor validation IoU
                metric_name='mean_iou',
                save_path=self.checkpoint_dir / 'best_model_early_stopping.pth',
                verbose=True
            )
        else:
            self.early_stopping = None

        # --- Resume Logic ---
        # Prioritize command line resume arg over config resume path
        resume_path_arg = self.config.get('resume_from') # From main() args parsing
        resume_path_cfg = self.train_config.get('checkpoint', {}).get('resume_path')
        resume_path = resume_path_arg if resume_path_arg else resume_path_cfg

        if resume_path and Path(resume_path).exists():
            self.load_checkpoint(resume_path)
            print(f"Resumed training from checkpoint: {resume_path}")
        elif self.train_config.get('checkpoint', {}).get('resume', False) and resume_path_cfg:
             # Fallback to config resume flag if resume_path exists
             if Path(resume_path_cfg).exists():
                 self.load_checkpoint(resume_path_cfg)
                 print(f"Resumed training from config checkpoint: {resume_path_cfg}")
             else:
                 print(f"Warning: Resume specified in config, but path '{resume_path_cfg}' not found.")
        else:
            print("Starting training from scratch.")

        # Store class colors for visualization if provided in config
        self.class_colors = self.config.get('class_colors', None)
        if self.class_colors:
             print(f"Loaded {len(self.class_colors)} class colors for visualization.")
        else:
             print("Warning: No 'class_colors' found in config for visualization.")


    def _log_model_info(self):
        """Log model architecture and parameter count for each component."""
        total_params = 0
        print("\n--- Model Architecture & Parameters ---")
        for name, component in self.model_components.items():
            if isinstance(component, nn.Module):
                params = sum(p.numel() for p in component.parameters() if p.requires_grad)
                print(f"  {name:<20}: {params:>12,} parameters")
                total_params += params
            else:
                 print(f"  {name:<20}: Not an nn.Module")
        print("-----------------------------------------")
        print(f"  {'Total Trainable':<20}: {total_params:>12,}")
        print("-----------------------------------------\n")

    def _create_model(self) -> Dict[str, nn.Module]:
        """Create model components based on config."""
        components = {}

        # === Camera Backbone ===
        cam_config = self.model_config.get('camera_backbone', {})
        components['camera_backbone'] = EfficientNetV2Backbone(
            pretrained=cam_config.get('pretrained', True)
        )
        cam_out_channels = components['camera_backbone'].out_channels

        # === LiDAR Backbone ===
        lidar_config = self.model_config.get('lidar_backbone', {})
        voxel_size = lidar_config.get('voxel_size', [0.4, 0.4, 0.8])
        point_cloud_range = lidar_config.get('point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        print(f"LiDAR Backbone using voxel_size: {voxel_size}, range: {point_cloud_range}")

        components['lidar_backbone'] = SECONDBackbone(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=lidar_config.get('max_num_points', 32),
            max_voxels=lidar_config.get('max_voxels', 15000)
        )

        # === Image BEV Lifter ===
        lifter_config = self.model_config.get('bev_lifter', {})
        bev_skip_channels = {
            'stage1': 32,  # Early features (highest resolution) - keep for fine detail
            'stage3': 128  # Deep features - keep for semantic info
            # Removed stage2 to reduce memory and compute as it's redundant
        }
        
        components['image_lifter'] = DepthAugmentedBEVLifter(
            img_channels_dict=cam_out_channels,
            bev_size=lifter_config.get('bev_size', (128, 128)),
            bev_skip_channels=bev_skip_channels,
            main_bev_channels=lifter_config.get('main_channels', 128),
            main_bev_source_stages=['stage5'],  # Use deepest features for main path
            voxel_size=tuple(voxel_size[:2])  # XY voxel sizes
        )
        
        # === Final LiDAR Feature Projection ===
        # Project final LiDAR BEV features to match main image BEV channels
        components['final_lidar_projection'] = nn.Sequential(
            nn.Conv2d(256, lifter_config.get('main_channels', 128), 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(lifter_config.get('main_channels', 128)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate * 0.5)
        )
        
        # === Fusion Module ===
        fusion_config = self.model_config.get('fusion', {})
        components['fusion'] = BEVFusion(
            lidar_channels=lifter_config.get('main_channels', 128),  # Projected final LiDAR channels
            image_channels=lifter_config.get('main_channels', 128),  # Main image BEV channels
            output_channels=fusion_config.get('output_channels', 256),
            spatial_size=lifter_config.get('bev_size', (128, 128)),
            chunk_size=fusion_config.get('chunk_size', 2048),  # Increased from default 1024
            use_reentrant=fusion_config.get('use_reentrant', True)  # Enable by default for better memory efficiency
        )
        
        # === Segmentation Head ===
        head_config = self.model_config.get('segmentation_head', {})
        loss_config = self.train_config.get('loss', {})
        
        components['head'] = BEVSegmentationHead(
            in_channels=fusion_config.get('output_channels', 256),
            skip_channels=bev_skip_channels,  # Must match lifter's skip channels
            decoder_channels=head_config.get('decoder_channels', (128, 64, 32)),
            num_classes=head_config.get('num_classes', 6),
            dropout=self.dropout_rate,
            use_focal_loss=loss_config.get('use_focal_loss', True),
            focal_gamma=loss_config.get('focal_loss', {}).get('gamma', 1.5),
            class_weights=loss_config.get('class_weights', [0.5, 0.5, 5.0, 5.0, 1.5, 0.75]),
            use_dice_loss=loss_config.get('use_dice_loss', True),
            dice_weight=loss_config.get('dice_weight', 0.5),
            dice_smooth=loss_config.get('dice_smooth', 1.0),
            label_smoothing=loss_config.get('label_smoothing', 0.05)
        )

        return components

    def train_epoch(self):
        """Train the model for one epoch."""
        # Set components to train mode
        for component in self.model_components.values():
            if isinstance(component, nn.Module):
                component.train()
        self.train_metrics.reset()

        epoch_start_time = time.time()
        running_loss = 0.0
        processed_samples = 0
        
        # Gradient accumulation setup
        accumulation_steps = 2 # Effective batch size = 4 * 2 = 8
        self.optimizer.zero_grad() # Zero gradients at the beginning of the epoch

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.config['epochs']}", unit="batch")
        torch.backends.cudnn.benchmark = True

        # Track visualizations per epoch
        num_visualizations = 0
        max_visualizations = 5  # Show 5 visualizations per epoch
        
        # Calculate visualization interval based on dataset size
        total_batches = len(self.train_loader)
        viz_interval = max(total_batches // max_visualizations, 1)
        
        print(f"Will visualize predictions approximately every {viz_interval} batches (Effective BS=8)") # Updated print

        for batch_idx, batch in enumerate(progress_bar):
            # Ensure required data is present
            if 'calib' not in batch:
                raise ValueError("Batch missing required 'calib' data for BEV lifting")

            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            points_list = [p.to(self.device, non_blocking=True) for p in batch['lidar']]
            targets = batch['bev_label'].to(self.device, non_blocking=True)
            
            # Use the helper function to move nested calibration data
            calib = move_to_device(batch['calib'], self.device)

            # Ensure targets are long type if they are class indices
            if targets.dim() == 3:
                targets = targets.long()
            elif targets.dim() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1).long()

            # --- Forward Pass ---
            # self.optimizer.zero_grad(set_to_none=True) # Moved outside loop

            with amp.autocast(enabled=self.scaler.is_enabled()):
                # 1. Camera Backbone & BEV Lifting
                image_feats_all_stages = self.camera_backbone(images)
                main_image_bev, image_bev_skips = self.image_lifter(
                    image_feats_all_stages,
                    calib
                )
                
                # 2. LiDAR Backbone & Final Projection
                lidar_feats = self.lidar_backbone(points_list)
                final_lidar_bev = lidar_feats['bev_features']['stage3']
                projected_lidar_bev = self.final_lidar_projection(final_lidar_bev)
                
                # 3. Fusion
                fused_features, _ = self.fusion(
                    projected_lidar_bev,
                    main_image_bev,
                    image_bev_skips
                )
                
                # 4. Head (with skip connections)
                losses = self.head(fused_features, image_bev_skips, targets)
                total_loss = losses['total_loss']
                
                # Normalize loss for accumulation
                total_loss = total_loss / accumulation_steps

            # --- Backward Pass & Optimization ---
            self.scaler.scale(total_loss).backward()
            
            # Perform optimizer step every accumulation_steps or on the last batch
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                # Optional gradient clipping (applied before optimizer step)
                if self.train_config.get('optimizer', {}).get('grad_clip'):
                    self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping
                    grad_clip_value = self.train_config['optimizer']['grad_clip']
                    torch.nn.utils.clip_grad_norm_(
                        [p for c in self.model_components.values() for p in c.parameters() if p.grad is not None], # Only clip params with grads
                        grad_clip_value
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True) # Zero gradients after step


            # --- Logging & Metrics ---
            batch_size = images.size(0)
            # Log loss before normalization for correct epoch average
            running_loss += losses['total_loss'].item() * batch_size 
            processed_samples += batch_size

            # Update metrics (consider frequency vs effective batch)
            # Maybe update metrics less often if accumulation step is large
            if batch_idx % 20 == 0: 
                with torch.no_grad():
                    predictions = self.head(fused_features, image_bev_skips)
                    self.train_metrics.update(predictions, targets)

                # Update progress bar (use non-normalized loss for display)
                lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}", # Show actual batch loss
                    'LR': f"{lr:.1e}"
                })
                
                # Visualize predictions at intervals to get 5 visualizations per epoch
                if num_visualizations < max_visualizations and (batch_idx % viz_interval == 0 or batch_idx == total_batches - 1) and self.class_colors is not None:
                    try:
                        # Get predictions for visualization
                        with torch.no_grad():
                            preds = self.head(fused_features, image_bev_skips)
                        
                        # Visualize with a unique identifier for this visualization in the epoch
                        viz_name = f'train_epoch{self.epoch+1}_viz{num_visualizations+1}'
                        self._visualize_predictions(images, preds, targets, prefix=viz_name)
                        num_visualizations += 1
                        print(f"Visualized training predictions {num_visualizations}/{max_visualizations} for epoch {self.epoch+1}")
                    except Exception as e:
                        print(f"Warning: Failed to visualize training predictions: {e}")

            self.train_step += 1 # Increment train step per actual batch

            # Clear cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        # --- End of Epoch ---
        epoch_time = time.time() - epoch_start_time
        avg_loss = running_loss / processed_samples if processed_samples > 0 else 0.0

        # Compute final epoch metrics
        epoch_metrics = self.train_metrics.get_metrics()
        epoch_metrics['loss'] = avg_loss

        # Log epoch summary
        print(f"\nEpoch {self.epoch+1} Train Summary:")
        print(f"  Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f} | Mean IoU: {epoch_metrics.get('mean_iou', 0):.4f}")
        if 'class_iou' in epoch_metrics:
            iou_str = " | ".join([f"{iou:.3f}" for iou in epoch_metrics['class_iou']])
            print(f"  Class IoU: [ {iou_str} ]")

        # Step LR scheduler (if epoch-based)
        if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        self.epoch += 1
        return epoch_metrics

    # --- Validation Method ---
    @torch.no_grad() # Decorator for no gradient calculation
    def validate(self):
        """Run validation on the validation dataset."""
        # Set components to eval mode
        for component in self.model_components.values():
            if isinstance(component, nn.Module):
                component.eval()
        self.val_metrics.reset()

        running_loss = 0.0
        processed_samples = 0
        val_start_time = time.time()

        progress_bar = tqdm(self.val_loader, desc=f"Validating Epoch {self.epoch}", unit="batch", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            # Ensure required data is present
            if 'calib' not in batch:
                raise ValueError("Batch missing required 'calib' data for BEV lifting")

            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            points_list = [p.to(self.device, non_blocking=True) for p in batch['lidar']]
            targets = batch['bev_label'].to(self.device, non_blocking=True)
            
            # Use the helper function to move nested calibration data
            calib = move_to_device(batch['calib'], self.device)

            # Ensure targets are long type if they are class indices
            if targets.dim() == 3:
                targets = targets.long()
            elif targets.dim() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1).long()

            # --- Forward Pass (same as training) ---
            # 1. Camera Backbone & BEV Lifting
            image_feats_all_stages = self.camera_backbone(images)
            main_image_bev, image_bev_skips = self.image_lifter(
                image_feats_all_stages,
                calib
            )
            
            # 2. LiDAR Backbone & Final Projection
            lidar_feats = self.lidar_backbone(points_list)
            final_lidar_bev = lidar_feats['bev_features']['stage3']
            projected_lidar_bev = self.final_lidar_projection(final_lidar_bev)
            
            # 3. Fusion
            fused_features, _ = self.fusion(
                projected_lidar_bev,
                main_image_bev,
                image_bev_skips
            )
            
            # 4. Head (with skip connections)
            if targets is not None:
                losses = self.head(fused_features, image_bev_skips, targets)
                total_loss = losses['total_loss']
            
            # Get predictions for metrics
            predictions = self.head(fused_features, image_bev_skips)

            # --- Update Metrics & Loss ---
            self.val_metrics.update(predictions, targets)

            batch_size = images.size(0)
            running_loss += total_loss.item() * batch_size
            processed_samples += batch_size

            if batch_idx % 20 == 0:
                progress_bar.set_postfix({'Loss': f"{total_loss.item():.4f}"})

            # Visualize predictions
            if batch_idx == 0 and self.epoch == 1 and self.class_colors is not None:
                try:
                    self._visualize_predictions(images, predictions, targets, prefix='val')
                except Exception as e:
                    print(f"Warning: Failed to visualize validation predictions: {e}")

        # --- End of Validation ---
        val_metrics = self.val_metrics.get_metrics()
        avg_loss = running_loss / processed_samples
        val_metrics['loss'] = avg_loss

        val_time = time.time() - val_start_time

        # Log validation summary
        print(f"\nEpoch {self.epoch} Validation Summary:")
        print(f"  Time: {val_time:.2f}s | Loss: {avg_loss:.4f} | Mean IoU: {val_metrics.get('mean_iou', 0):.4f}")
        if 'class_iou' in val_metrics:
            iou_str = " | ".join([f"{iou:.3f}" for iou in val_metrics['class_iou']])
            print(f"  Class IoU: [ {iou_str} ]")

        # Step scheduler if ReduceLROnPlateau
        if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_metrics.get('mean_iou', 0.0))

        return val_metrics

    def _visualize_predictions(self, images: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor, prefix: str = 'train'):
        """Visualize input image, ground truth, and prediction."""
        if self.class_colors is None:
            print("Skipping visualization: class_colors not defined.")
            return

        num_samples_to_show = min(4, images.size(0)) # Show max 4 samples
        preds_classes = predictions.argmax(dim=1) # Get predicted class indices [B, H, W]

        # Ensure targets are class indices [B, H, W]
        if targets.dim() == 4:
            if targets.size(1) == 1: # [B, 1, H, W] indices
                targets_classes = targets.squeeze(1)
            else: # [B, C, H, W] one-hot
                targets_classes = targets.argmax(dim=1)
        else: # Already [B, H, W]
            targets_classes = targets

        fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(15, 5 * num_samples_to_show))
        if num_samples_to_show == 1: axes = axes.reshape(1, -1) # Ensure axes is 2D

        class_names = self.config.get('class_names', [f'Class {i}' for i in range(len(self.class_colors))]) # Get class names if available

        for i in range(num_samples_to_show):
            # --- Image ---
            # Assuming image is channel-first, float, normalized [0,1] or similar
            img_np = images[i].cpu().permute(1, 2, 0).numpy()
            # Simple unnormalization heuristic if needed (adjust based on your actual normalization)
            img_np = np.clip(img_np * 0.5 + 0.5, 0, 1) if img_np.min() < 0 else img_np
            img_np = (img_np * 255).astype(np.uint8)

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Sample {i}: Input Image")
            axes[i, 0].axis('off')

            # --- Ground Truth ---
            target_map = targets_classes[i].cpu().numpy()
            target_rgb = np.zeros((target_map.shape[0], target_map.shape[1], 3), dtype=np.uint8)
            for cls_idx, color in enumerate(self.class_colors):
                target_rgb[target_map == cls_idx] = color
            axes[i, 1].imshow(target_rgb)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')

            # --- Prediction ---
            pred_map = preds_classes[i].cpu().numpy()
            pred_rgb = np.zeros((pred_map.shape[0], pred_map.shape[1], 3), dtype=np.uint8)
            for cls_idx, color in enumerate(self.class_colors):
                pred_rgb[pred_map == cls_idx] = color
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')

        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=np.array(c)/255.0) for c in self.class_colors]
        fig.legend(handles, class_names, loc='lower center', ncol=min(6, len(class_names)), bbox_to_anchor=(0.5, 0.01))
        plt.tight_layout(rect=[0, 0.05, 1, 0.98]) # Adjust layout for legend

        # Save figure using the full prefix which includes visualization number
        viz_path = self.viz_dir / f'{prefix}_predictions.png' # Use prefix directly
        plt.savefig(viz_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization to {viz_path}")

    def _plot_confusion_matrix(self, cm: np.ndarray, prefix: str = ''):
        """Plot confusion matrix."""
        if cm is None or not isinstance(cm, np.ndarray): return

        try:
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(10, 8))
            class_names = self.config.get('class_names', [f'Class {i}' for i in range(cm.shape[0])])
            # Normalize CM row-wise (shows percentage of true class predicted as X)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm) # Handle potential division by zero if a class has no samples

            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                        xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'{prefix.capitalize()} Confusion Matrix (Epoch {self.epoch}) - Normalized by True Class')
            plt.tight_layout()
            cm_path = self.viz_dir / f'{prefix}_confusion_matrix_epoch{self.epoch}.png'
            plt.savefig(cm_path)
            plt.close(fig)
            print(f"Saved confusion matrix to {cm_path}")
        except ImportError:
            print("Warning: Seaborn not installed. Cannot plot confusion matrix.")
        except Exception as e:
            print(f"Warning: Failed to plot confusion matrix: {e}")


    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save model checkpoint including all components."""
        if path is None:
            path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
        else:
            path = Path(path)

        # Prepare checkpoint data for all components
        checkpoint = {
            'epoch': self.epoch,
            'train_step': self.train_step,
            'best_val_iou': self.best_val_iou,
            # Save state dicts for all model components
            'camera_backbone_state_dict': self.camera_backbone.state_dict(),
            'lidar_backbone_state_dict': self.lidar_backbone.state_dict(),
            'image_lifter_state_dict': self.image_lifter.state_dict(),
            'final_lidar_projection_state_dict': self.final_lidar_projection.state_dict(),
            'fusion_state_dict': self.fusion.state_dict(),
            'head_state_dict': self.head.state_dict(),
            # Save optimizer, scheduler, scaler state
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            # Include configs for reference
            'model_config': self.model_config,
            'train_config': self.train_config
        }

        # Save checkpoint
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

        # Handle 'best' checkpoint saving
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            shutil.copyfile(path, best_path)
            print(f"Best model checkpoint updated at {best_path}")

        # Save last checkpoint symlink or copy
        last_path = self.checkpoint_dir / 'last_model.pth'
        if last_path.is_symlink() or last_path.exists():
            last_path.unlink()
        try:
            relative_path = path.relative_to(self.checkpoint_dir)
            last_path.symlink_to(relative_path)
        except OSError:
            shutil.copyfile(path, last_path)
        print(f"Last model checkpoint updated at {last_path}")


    def load_checkpoint(self, path: str):
        """Load checkpoint for all components."""
        path = Path(path)
        if not path.exists():
            print(f"Error: Checkpoint path not found: {path}")
            return False

        print(f"Loading checkpoint from {path}...")
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Load model weights for all components
            self.camera_backbone.load_state_dict(checkpoint['camera_backbone_state_dict'])
            self.lidar_backbone.load_state_dict(checkpoint['lidar_backbone_state_dict'])
            self.image_lifter.load_state_dict(checkpoint['image_lifter_state_dict'])
            self.final_lidar_projection.load_state_dict(checkpoint['final_lidar_projection_state_dict'])
            self.fusion.load_state_dict(checkpoint['fusion_state_dict'])
            self.head.load_state_dict(checkpoint['head_state_dict'])

            # Load optimizer, scheduler, scaler state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # Load training state
            self.epoch = checkpoint.get('epoch', 0) + 1
            self.train_step = checkpoint.get('train_step', 0)
            self.best_val_iou = checkpoint.get('best_val_iou', 0.0)

            print(f"Successfully loaded checkpoint. Resuming from Epoch {self.epoch}.")
            return True

        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Remove unused methods like export, profile etc. if not needed

# --- Custom Collate Function ---
def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized point clouds and nested calibration data."""
    images, lidar_points, bev_labels, sample_tokens, scene_tokens = [], [], [], [], []
    
    # For calibration data
    all_calibs = []

    for item in batch:
        images.append(item['image'])
        lidar_points.append(item['lidar']) # Keep as list
        label = item['bev_label']
        # Standardize BEV label shape to [1, H, W] or [C, H, W]
        if label.dim() == 2: label = label.unsqueeze(0) # [H, W] -> [1, H, W]
        bev_labels.append(label)
        
        # Add other items if they exist
        if 'sample_token' in item: sample_tokens.append(item['sample_token'])
        if 'scene_token' in item: scene_tokens.append(item['scene_token'])
        
        # Simply collect calibration dictionaries - will handle merging after
        if 'calib' in item:
            all_calibs.append(item['calib'])

    batch_dict = {
        'image': torch.stack(images),
        # Revert to passing LiDAR points as a list
        'lidar': lidar_points, 
        'bev_label': torch.stack(bev_labels)
    }
    
    # Process calibration data recursively
    if all_calibs:
        batch_dict['calib'] = collate_nested_dict(all_calibs)
        
    # Add optional items
    if sample_tokens: batch_dict['sample_token'] = sample_tokens
    if scene_tokens: batch_dict['scene_token'] = scene_tokens

    return batch_dict

def collate_nested_dict(batch_dicts):
    """Recursively collate nested dictionaries."""
    if not batch_dicts:
        return {}
    
    # Check the structure of the first dictionary to determine how to collate
    sample_dict = batch_dicts[0]
    result = {}
    
    for key in sample_dict:
        # Get values for this key from all dictionaries
        values = [d[key] for d in batch_dicts if key in d]
        
        # Check the type of the first value to determine how to collate
        if not values:
            continue
            
        sample_value = values[0]
        
        if isinstance(sample_value, torch.Tensor):
            # Stack tensors along a new dimension
            result[key] = torch.stack(values, dim=0)
        elif isinstance(sample_value, dict):
            # Recursively collate dictionaries
            result[key] = collate_nested_dict(values)
        elif isinstance(sample_value, list):
            # For lists of tensors or other objects
            if values and all(isinstance(v, torch.Tensor) for v in values[0]):
                # List of tensors - stack each position separately
                result[key] = [torch.stack([batch[i] for batch in values], dim=0) 
                              for i in range(len(values[0]))]
            else:
                # Other types of lists - just concatenate
                result[key] = values
        else:
            # For other types (scalars, strings, etc.) - just put in a list
            result[key] = values
            
    return result

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Train BEV Fusion model')
    # Keep existing args...
    parser.add_argument('--model-config', type=str, required=True, help='Path to model configuration YAML')
    parser.add_argument('--train-config', type=str, required=True, help='Path to training configuration YAML')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to dataset (e.g., NuScenes)')
    parser.add_argument('--bev-labels-dir', type=str, required=True, help='Path to precomputed BEV labels')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (overrides config)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with smaller dataset')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--num-workers', type=int, default=None, help='Override number of workers from config')
    parser.add_argument('--optimize-memory', action='store_true', help='Enable memory optimization techniques')
    parser.add_argument('--image-size', type=str, default=None, help='Override image size W,H (e.g., "896,512")')
    parser.add_argument('--early-stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping (default: 7)')
    parser.add_argument('--weight-decay', type=float, default=None, help='Override weight decay')
    parser.add_argument('--dropout', type=float, default=None, help='Override dropout rate')
    parser.add_argument('--augmentation-strength', type=str, default='medium', choices=['none', 'light', 'medium', 'strong'], help='Strength of data augmentation')
    # Add other args as needed

    args = parser.parse_args()

    # --- Load Configs ---
    try:
        with open(args.model_config, 'r') as f: model_config = yaml.safe_load(f)
        with open(args.train_config, 'r') as f: train_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Error: Config file not found - {e}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return

    # --- Output Dir & Config Backup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    try:
        shutil.copy(args.model_config, run_dir / Path(args.model_config).name)
        shutil.copy(args.train_config, run_dir / Path(args.train_config).name)
        print(f"Outputs will be saved to: {run_dir}")
    except Exception as e:
        print(f"Warning: Could not copy config files to output directory: {e}")

    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        # Set benchmark mode if input sizes are constant
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for faster matmuls on Ampere+ GPUs (PyTorch >= 1.7)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Override Configs from Args ---
    if args.batch_size is not None: train_config['train']['batch_size'] = args.batch_size
    if args.num_workers is not None: train_config['train']['num_workers'] = args.num_workers
    if args.weight_decay is not None: train_config['optimizer']['weight_decay'] = args.weight_decay
    # Dropout override needs to be passed to Trainer init
    dropout_override = args.dropout if args.dropout is not None else model_config.get('segmentation_head', {}).get('dropout', 0.2)

    # Image size override
    if args.image_size:
        try:
            width, height = map(int, args.image_size.split(','))
            # Assuming model_config structure - adjust if needed
            if 'input_config' not in model_config: model_config['input_config'] = {}
            model_config['input_config']['image_size'] = [height, width]
            print(f"Overriding image size to {width}x{height}")
        except ValueError:
            print(f"Error: Invalid image size format '{args.image_size}'. Use W,H (e.g., '896,512'). Using config default.")
    # Get final image size for transform
    img_h, img_w = model_config.get('input_config', {}).get('image_size', [376, 672]) # Default if not set


    # --- Dataset & Dataloader Setup ---
    print("Creating datasets and dataloaders...")
    # Image transform (consider moving normalization inside model or dataset)
    # Basic resize transform
    def image_transform(image_np):
        resized_image = cv2.resize(image_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR) # Use INTER_LINEAR
        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1) / 255.0 # Basic normalization
        # Add proper normalization if EfficientNet expects it (usually ImageNet stats)
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # tensor_image = normalize(tensor_image)
        return tensor_image

    # Augmentation setup - Optimized for thin line segmentation based on recommendations
    augmentor = None
    if args.augmentation_strength != 'none':
        print(f"Using {args.augmentation_strength} data augmentation optimized for thin line segmentation.")
        # Define BEVAugmentor based on strength with focus on most effective augmentations
        if args.augmentation_strength == 'light':
            augmentor = BEVAugmentor(
                flip_prob=0.3,                    # Keep horizontal flip (effective)
                rotate_prob=0.3,                  # Small rotation probability
                rotate_range=(-3, 3),             # Very small angle range (preserve alignment)
                scale_prob=0.0,                   # Disable scaling (can break alignment)
                translate_prob=0.0,               # Disable translation (can break alignment)
                cutout_prob=0.3,                  # Enable cutout for better contextual learning
                cutout_max_boxes=2,               # Fewer cutout regions
                cutout_max_size=(20, 20),         # Moderate size cutouts
                mixup_prob=0.0,                   # Disable mixup (complex, can create unrealistic scenarios)
                feature_noise_prob=0.3,           # Add noise to LiDAR points (simulate sensor noise)
                feature_noise_std=0.01,           # Small noise for LiDAR points
                color_jitter_prob=0.3             # Add color jitter to improve robustness to lighting
            )
        elif args.augmentation_strength == 'medium':
            augmentor = BEVAugmentor(
                flip_prob=0.5,                    # Keep horizontal flip
                rotate_prob=0.5,                  # Moderate rotation probability  
                rotate_range=(-5, 5),             # Still small angle range but slightly larger
                scale_prob=0.0,                   # Disable scaling
                translate_prob=0.0,               # Disable translation
                cutout_prob=0.5,                  # Increase cutout probability
                cutout_max_boxes=2,               # Use 2 cutout regions
                cutout_max_size=(25, 25),         # Slightly larger cutouts
                mixup_prob=0.0,                   # Disable mixup
                feature_noise_prob=0.7,           # Increase noise probability
                feature_noise_std=0.015,          # Slightly more noise
                color_jitter_prob=0.7             # More color jittering
            )
        elif args.augmentation_strength == 'strong':
            augmentor = BEVAugmentor(
                flip_prob=0.5,                    # Keep horizontal flip
                rotate_prob=0.7,                  # Higher rotation probability
                rotate_range=(-5, 5),             # Keep small angle range for alignment
                scale_prob=0.0,                   # Disable scaling
                translate_prob=0.0,               # Disable translation
                cutout_prob=0.7,                  # High cutout probability
                cutout_max_boxes=3,               # More cutout regions
                cutout_max_size=(30, 30),         # Larger cutouts
                mixup_prob=0.0,                   # Disable mixup
                feature_noise_prob=0.9,           # Very high noise probability
                feature_noise_std=0.02,           # More noise for greater robustness
                color_jitter_prob=0.9             # Extensive color jittering
            )
        
        # Add custom LiDAR point dropout function to the augmentor
        def lidar_point_dropout(batch, dropout_prob=0.15):
            """Apply random point dropout to LiDAR points to simulate occlusions and variable density"""
            if 'lidar' in batch:
                for i, points in enumerate(batch['lidar']):
                    if len(points) > 0:
                        # Calculate how many points to keep
                        keep_ratio = 1.0 - dropout_prob
                        num_keep = max(int(len(points) * keep_ratio), 10)  # Keep at least 10 points
                        
                        # Randomly select indices to keep
                        keep_indices = torch.randperm(len(points))[:num_keep]
                        batch['lidar'][i] = points[keep_indices]
            return batch
            
        # Monkey-patch the augmentor with the new method
        augmentor.lidar_point_dropout = lidar_point_dropout
        
        # Store the original __call__ method
        original_call = augmentor.__call__
        
        # Create a new __call__ method that also applies point dropout
        def enhanced_call(batch, lidar_only=False, mixup_batch=None):
            # Apply original augmentations
            batch = original_call(batch, lidar_only, mixup_batch)
            
            # Apply point dropout with 70% probability
            if random.random() < 0.7:
                dropout_prob = random.uniform(0.05, 0.25)  # Random dropout percentage
                batch = lidar_point_dropout(batch, dropout_prob)
                
            return batch
        
        # Replace the __call__ method
        augmentor.__call__ = enhanced_call
        
        print(f"Added LiDAR point dropout augmentation to simulate occlusions and variable density.")

    # Create datasets
    try:
        base_train_dataset = PrecomputedBEVDataset(
            dataroot=args.dataroot, bev_labels_dir=args.bev_labels_dir, split='train',
            transform=image_transform # Apply resize transform
        )
        val_dataset = PrecomputedBEVDataset(
            dataroot=args.dataroot, bev_labels_dir=args.bev_labels_dir, split='val',
            transform=image_transform
        )
    except FileNotFoundError as e:
         print(f"Error creating dataset: {e}. Check dataroot and bev_labels_dir paths.")
         return

    # Apply augmentation wrapper
    train_dataset = RegularizedTrainingWrapper(base_train_dataset, augmentor=augmentor) if augmentor else base_train_dataset

    # Debug subset
    if args.debug:
        print("DEBUG MODE: Using subset of 100 samples.")
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(8530, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(1706, len(val_dataset))))

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Create dataloaders
    bs = train_config['train']['batch_size']
    nw = train_config['train']['num_workers']
    pm = train_config['train'].get('pin_memory', True)
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=pm, collate_fn=custom_collate_fn, drop_last=True,
        persistent_workers=nw > 0 # Use persistent workers if num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config['val']['batch_size'], shuffle=False, num_workers=nw,
        pin_memory=pm, collate_fn=custom_collate_fn,
        persistent_workers=nw > 0
    )

    # --- Create Combined Config for Trainer ---
    # (Trainer now takes model_config and train_config separately)
    # Create a flat config dict mainly for logging/reference if needed
    combined_config = model_config.copy()
    combined_config.update(train_config)
    combined_config['epochs'] = train_config['train']['epochs'] # Ensure epochs is top-level
    combined_config['resume_from'] = args.resume # Pass resume path from args

    # --- Initialize Trainer ---
    trainer = Trainer(
        config=combined_config, # Pass combined for reference, though split configs are used internally
        train_config=train_config,
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=run_dir, # Save outputs to the timestamped run directory
        use_early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
        # Pass overrides
        weight_decay=train_config['optimizer']['weight_decay'],
        dropout_rate=dropout_override
    )

    # --- Training Loop ---
    print(f"\n--- Starting Training (Run: {run_dir.name}) ---")
    start_time = time.time()

    try:
        for epoch in range(trainer.epoch, train_config['train']['epochs']):
            print(f"\n>>> Epoch {epoch+1}/{train_config['train']['epochs']}")
            # Train
            train_metrics = trainer.train_epoch()

            # Validate
            val_metrics = None
            if (epoch + 1) % train_config['val']['interval'] == 0:
                val_metrics = trainer.validate()

                current_val_iou = val_metrics.get('mean_iou', 0.0)

                # Checkpointing - Best model based on Val IoU
                if current_val_iou > trainer.best_val_iou:
                    print(f"  New best validation IoU: {current_val_iou:.4f} (prev: {trainer.best_val_iou:.4f})")
                    trainer.best_val_iou = current_val_iou
                    trainer.save_checkpoint(is_best=True) # Saves best_model.pth
                else:
                     print(f"  Validation IoU: {current_val_iou:.4f} (Best: {trainer.best_val_iou:.4f})")


                # Early stopping check
                if trainer.early_stopping:
                    trainer.early_stopping(current_val_iou, trainer) # Pass score and trainer to save model
                    if trainer.early_stopping.early_stop:
                        print(f"Early stopping triggered at epoch {epoch+1}.")
                        break

            # Regular checkpoint saving (independent of validation or best model)
            if (epoch + 1) % train_config['checkpoint'].get('save_interval', 5) == 0:
                # Pass specific path to save_checkpoint for regular epoch saves
                 epoch_checkpoint_path = trainer.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                 trainer.save_checkpoint(path=epoch_checkpoint_path, is_best=False) # is_best=False for regular saves


            # Clear cache between epochs
            if device.type == 'cuda': torch.cuda.empty_cache()

            # Print memory stats (optional)
            if device.type == 'cuda':
                print(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB allocated, "
                      f"{torch.cuda.memory_reserved(0) / 1024**2:.1f}MB reserved")


        # --- Training Finished ---
        total_time = time.time() - start_time
        print(f"\n--- Training Finished ---")
        print(f"Total Time: {datetime.timedelta(seconds=total_time)}")
        print(f"Best Validation Mean IoU: {trainer.best_val_iou:.4f}")
        if trainer.early_stopping and trainer.early_stopping.early_stop:
             print(f"Stopped early at epoch {trainer.early_stopping.stopped_epoch} (Best score at epoch {trainer.early_stopping.best_epoch})")
        print(f"Checkpoints and logs saved in: {run_dir}")

    except KeyboardInterrupt:
        print("\n--- Training Interrupted ---")
        # Save checkpoint on interrupt
        trainer.save_checkpoint(path=trainer.checkpoint_dir / 'interrupt_checkpoint.pth')
    except Exception as e:
        print(f"\n--- Error During Training ---")
        import traceback
        traceback.print_exc()
        # Optionally save checkpoint on other errors too
        # trainer.save_checkpoint(path=trainer.checkpoint_dir / 'error_checkpoint.pth')
    finally:
         # Ensure any open resources are closed (e.g., TensorBoard writer if used)
         pass


if __name__ == '__main__':
    main()