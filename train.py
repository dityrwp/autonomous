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
import matplotlib.pyplot as plt
import logging

from models.backbones import EfficientNetV2Backbone, SECONDBackbone
from models.fusion import BEVFusion
from models.heads import BEVSegmentationHead
from datasets.precomputed_bev_dataset import PrecomputedBEVDataset
from datasets.augmentation import BEVAugmentor, RegularizedTrainingWrapper
from utils.metrics import SegmentationMetrics, MetricsLogger
from utils.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Trainer:
    def __init__(
        self,
        config: dict,
        train_config: dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = None,
        output_dir: Path = None,
        use_early_stopping: bool = True,
        early_stopping_patience: int = 5,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize trainer.
        
        Args:
            config: Model configuration dictionary
            train_config: Training configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training
            output_dir: Output directory
            use_early_stopping: Whether to use early stopping
            early_stopping_patience: Patience for early stopping
            weight_decay: Weight decay for regularization
            dropout_rate: Dropout rate for regularization
        """
        # Store configuration
        self.config = config
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir) if output_dir else Path('outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup checkpoint directory
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup visualization directory
        self.viz_dir = self.output_dir / 'visualizationsz'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visualizations will be saved to {self.viz_dir}")
        
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
        
        # Use provided weight decay or fallback to config
        self.weight_decay = weight_decay or self.opt_config.get('weight_decay', 1e-4)
        
        # Use provided dropout or fallback to config
        self.dropout_rate = dropout_rate
        
        # Override dropout in model config
        if 'segmentation_head' in config:
            config['segmentation_head']['dropout'] = self.dropout_rate
        else:
            # For backward compatibility
            config['dropout'] = self.dropout_rate
        
        # Set up model and optimizer
        self.model = self._create_model()
        print(f"Initialized model on {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Print model parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {total_params:,} trainable parameters")
        
        # Metrics
        self.train_metrics = SegmentationMetrics(
            num_classes=config.get('num_classes', 6),
            device=self.device
        )
        
        self.val_metrics = SegmentationMetrics(
            num_classes=config.get('num_classes', 6),
            device=self.device
        )
        
        # Logging and progress tracking
        self.metrics_logger = MetricsLogger()
        
        # Setup optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW([
            {'params': self.camera_backbone.parameters(), 'lr': self.lr_camera},
            {'params': self.lidar_backbone.parameters(), 'lr': self.lr_lidar},
            {'params': self.fusion.parameters(), 'lr': self.lr_fusion},
            {'params': self.head.parameters(), 'lr': self.lr_head},
            {'params': self.image_projections.parameters(), 'lr': self.lr_fusion},
            {'params': self.lidar_projections.parameters(), 'lr': self.lr_fusion}
        ], weight_decay=self.weight_decay)
        
        # Setup scheduler
        scheduler_name = self.scheduler_config.get('name', 'CosineAnnealingLR')
        
        if scheduler_name == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config.get('train', {}).get('epochs', 50),
                eta_min=self.scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            # Reduce on plateau with patience
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
                mode='max',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 3),
                min_lr=self.scheduler_config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None
        
        # AMP scaler for mixed precision training
        self.scaler = amp.GradScaler()
        
        # Metrics tracking
        self.best_val_iou = 0.0
        self.train_step = 0
        self.epoch = 0
        
        # Early stopping setup
        self.use_early_stopping = use_early_stopping
        if use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                mode='max',
                metric_name='mean_iou',
                save_path=self.checkpoint_dir / 'best_model_early_stopping.pth',
                verbose=True
            )
        
        # Resume from checkpoint if specified
        resume_path = self.train_config.get('checkpoint', {}).get('resume_path')
        if resume_path and Path(resume_path).exists():
            self.load_checkpoint(resume_path)
            print(f"Resumed training from checkpoint: {resume_path}")
        
        # Log model architecture and parameters
        self._log_model_info()
    
    def _log_model_info(self):
        """Log model architecture and parameter count"""
        total_params = 0
        model_components = {
            'camera_backbone': self.camera_backbone,
            'lidar_backbone': self.lidar_backbone,
            'fusion': self.fusion,
            'head': self.head,
            'image_projections': self.image_projections,
            'lidar_projections': self.lidar_projections
        }
        
        component_params = {}
        for name, component in model_components.items():
            params = sum(p.numel() for p in component.parameters() if p.requires_grad)
            total_params += params
            component_params[name] = params
            
        # Log to console
        print("\nModel Architecture:")
        for name, params in component_params.items():
            print(f"  {name}: {params:,} parameters ({params / total_params * 100:.1f}%)")
        print(f"Total trainable parameters: {total_params:,}")
    
    def _create_model(self):
        """Create model components based on config"""
        # Initialize camera backbone
        self.camera_backbone = EfficientNetV2Backbone(
            pretrained=self.config.get('camera_backbone', {}).get('pretrained', True)
        ).to(self.device)
        
        # Initialize lidar backbone
        self.lidar_backbone = SECONDBackbone(
            voxel_size=self.config.get('lidar_backbone', {}).get('voxel_size', [0.8, 0.8, 0.8]),
            point_cloud_range=self.config.get('lidar_backbone', {}).get('point_cloud_range', 
                                                                  [-51.2, -51.2, -5, 51.2, 51.2, 3]),
            max_num_points=self.config.get('lidar_backbone', {}).get('max_num_points', 32),
            max_voxels=self.config.get('lidar_backbone', {}).get('max_voxels', 20000)
        ).to(self.device)
        
        # Define stage channels based on actual feature dimensions
        stage_channels = {
            'stage1': {'lidar': 64, 'image': 32},   # stage3 -> stage1
            'stage2': {'lidar': 128, 'image': 64},  # stage4 -> stage2
            'stage3': {'lidar': 256, 'image': 128}, # stage5 -> stage3
        }
        
        # Create projection layers for adapting feature dimensions
        self.image_projections = nn.ModuleDict()
        self.lidar_projections = nn.ModuleDict()
        
        # For image features: stage3->32, stage4->64, stage5->128 channels
        self.image_projections['stage1'] = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate / 2)
        ).to(self.device)
        
        self.image_projections['stage2'] = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate / 2)
        ).to(self.device)
        
        self.image_projections['stage3'] = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate / 2)
        ).to(self.device)
        
        # For lidar features
        self.lidar_projections['stage1'] = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate / 2)
        ).to(self.device)
        
        self.lidar_projections['stage2'] = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate / 2)
        ).to(self.device)
        
        self.lidar_projections['stage3'] = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate / 2)
        ).to(self.device)
        
        # Map config stages to fusion stages
        self.config_to_fusion_stage = {
            'stage3': 'stage1',
            'stage4': 'stage2',
            'stage5': 'stage3'
        }
        
        # Initialize fusion module
        fusion_config = self.config.get('fusion', {})
        self.fusion = BEVFusion(
            lidar_channels=fusion_config.get('lidar_channels', 128),
            image_channels=fusion_config.get('image_channels', {
                'stage3': 256,
                'stage4': 384,
                'stage5': 512
            }),
            output_channels=fusion_config.get('output_channels', 128),
            spatial_size=fusion_config.get('spatial_size', (128, 128)),
            stage_channels=stage_channels
        ).to(self.device)
        
        # Initialize segmentation head with updated parameters
        head_config = self.config.get('segmentation_head', {})
        
        # Get class colors for visualization
        class_colors = self.config.get('class_colors', [
            [252, 252, 252],  # Background
            [166, 206, 227],  # Drivable Area
            [202, 178, 214],  # Road Divider
            [106, 61, 154],   # Lane Divider
            [224, 74, 76],    # Walkway
            [251, 154, 153]   # Pedestrian Crossing
        ])
        
        # Use class weights from config or use defaults
        class_weights = head_config.get('class_weights', [0.75, 0.5, 5.0, 5.0, 2.0, 1.0])
        
        self.head = BEVSegmentationHead(
            in_channels=head_config.get('in_channels', 128),
            hidden_channels=head_config.get('hidden_channels', 128),
            num_classes=head_config.get('num_classes', 6),
            dropout=self.dropout_rate,  # Use our dropout rate
            use_focal_loss=head_config.get('use_focal_loss', True),
            focal_gamma=head_config.get('focal_gamma', 2.0),
            class_weights=class_weights,
            label_smoothing=head_config.get('label_smoothing', 0.05),
            use_dice_loss=head_config.get('use_dice_loss', True),
            dice_weight=head_config.get('dice_weight', 0.5),
            dice_smooth=head_config.get('dice_smooth', 1.0)
        ).to(self.device)
        
        # Return the main model component for early stopping
        return self.head
    
    def train_epoch(self):
        self.camera_backbone.train()
        self.lidar_backbone.train()
        self.fusion.train()
        self.head.train()
        self.train_metrics.reset()
        
        # Initialize progress tracking
        epoch_start_time = time.time()
        running_loss = 0.0
        processed_samples = 0
        
        # Create progress bar
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.config['epochs']}", 
                           unit="batch")
        
        # Calculate visualization interval - show 5 visualizations per epoch
        train_viz_interval = max(1, len(self.train_loader) // 5)
        print(f"Will visualize training predictions every {train_viz_interval} batches (5 times per epoch)")
        
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device (non-blocking for better performance)
            images = batch['image'].to(self.device, non_blocking=True)
            
            # Handle lidar data
            if isinstance(batch['lidar'], list):
                points_list = [points.to(self.device, non_blocking=True) for points in batch['lidar']]
            else:
                points_list = batch['lidar'].to(self.device, non_blocking=True)
            
            targets = batch['bev_label'].to(self.device, non_blocking=True)
            
            # Ensure targets are in the correct format
            # If targets are [B, 1, H, W] with class indices, convert to [B, H, W]
            if targets.dim() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            
            # Forward pass with AMP
            with amp.autocast():
                # Extract features from camera
                image_feats = self.camera_backbone(images)
                
                # Process each point cloud in the batch
                lidar_feats_list = []
                for points in points_list:
                    # Add batch dimension if missing
                    if points.dim() == 2:
                        points = points.unsqueeze(0)
                    lidar_feats = self.lidar_backbone(points)
                    lidar_feats_list.append(lidar_feats)
                
                # Stack lidar features if they have same shape, otherwise process individually
                if all(feat['bev_features']['stage3'].shape == lidar_feats_list[0]['bev_features']['stage3'].shape for feat in lidar_feats_list):
                    lidar_feats = {
                        'stage1': torch.cat([feat['bev_features']['stage1'] for feat in lidar_feats_list], dim=0),
                        'stage2': torch.cat([feat['bev_features']['stage2'] for feat in lidar_feats_list], dim=0),
                        'stage3': torch.cat([feat['bev_features']['stage3'] for feat in lidar_feats_list], dim=0)
                    }
                else:
                    # Process individually and combine later (placeholder)
                    lidar_feats = {
                        'stage1': lidar_feats_list[0]['bev_features']['stage1'],
                        'stage2': lidar_feats_list[0]['bev_features']['stage2'],
                        'stage3': lidar_feats_list[0]['bev_features']['stage3']
                    }
                
                # Create dictionary of lidar features for each stage with projections
                lidar_features_dict = {}
                for lidar_stage, fusion_stage in zip(['stage1', 'stage2', 'stage3'], ['stage1', 'stage2', 'stage3']):
                    lidar_features_dict[fusion_stage] = self.lidar_projections[fusion_stage](lidar_feats[lidar_stage])
                
                # Create dictionary of image features for each stage with projections
                image_features_dict = {}
                for config_stage, fusion_stage in self.config_to_fusion_stage.items():
                    if config_stage in image_feats:
                        image_features_dict[fusion_stage] = self.image_projections[fusion_stage](image_feats[config_stage])
                    elif 'stage5' in image_feats:  # Fallback to stage5 if needed
                        image_features_dict[fusion_stage] = self.image_projections[fusion_stage](image_feats['stage5'])
                
                # Fuse features
                fused_feats, _ = self.fusion(lidar_features_dict, image_features_dict)
                
                # Predict and compute loss
                losses = self.head(fused_feats, targets)
                
                # Get predictions for metrics - need to extract logits only
                with torch.no_grad():
                    # Call head without targets to get logits directly
                    predictions = self.head(fused_feats)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            
            # Check if loss is a tensor with grad_fn
            if isinstance(losses['total_loss'], torch.Tensor) and losses['total_loss'].requires_grad:
                self.scaler.scale(losses['total_loss']).backward()
                
                # Apply gradient clipping if configured
                grad_clip = self.opt_config.get('grad_clip', None)
                if grad_clip is not None:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    
                    # Also clip specific components (fusion might need stricter clipping)
                    torch.nn.utils.clip_grad_norm_(self.fusion.parameters(), grad_clip * 0.5)
                    torch.nn.utils.clip_grad_norm_(self.head.parameters(), grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Step OneCycleLR scheduler after each batch
                if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
            else:
                print(f"Warning: Loss is not a tensor with grad_fn. Type: {type(losses['total_loss'])}, "
                      f"requires_grad: {getattr(losses['total_loss'], 'requires_grad', None)}, "
                      f"grad_fn: {getattr(losses['total_loss'], 'grad_fn', None)}")
                # Create a dummy loss to avoid breaking the training loop
                dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                self.scaler.scale(dummy_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            # Update metrics (less frequently to reduce overhead)
            if batch_idx % 50 == 0:
                self.metrics_logger.update_loss(losses['total_loss'].item() if isinstance(losses['total_loss'], torch.Tensor) else 0.0)
                
                # Handle shape mismatch for metrics update
                if targets.dim() == 3:  # If targets are class indices [B, H, W]
                    # Ensure predictions is a 4D tensor with shape [B, C, H, W]
                    if isinstance(predictions, dict):
                        # If predictions is a dictionary (during training), extract logits
                        pred_tensor = predictions['logits'].detach() if 'logits' in predictions else predictions.detach()
                    else:
                        # If predictions is already a tensor
                        pred_tensor = predictions.detach()
                    
                    # Verify we have a 4D tensor before passing to metrics
                    if pred_tensor.dim() == 4:
                        self.train_metrics.update(pred_tensor, targets)
                    else:
                        print(f"Warning: Expected 4D predictions tensor, got shape {pred_tensor.shape}")
                else:
                    # Targets are already in correct format
                    if isinstance(predictions, dict):
                        # If predictions is a dictionary (during training), extract logits
                        pred_tensor = predictions['logits'].detach() if 'logits' in predictions else predictions.detach()
                    else:
                        # If predictions is already a tensor
                        pred_tensor = predictions.detach()
                    
                    # Verify we have a 4D tensor before passing to metrics
                    if pred_tensor.dim() == 4:
                        self.train_metrics.update(pred_tensor, targets)
                    else:
                        print(f"Warning: Expected 4D predictions tensor, got shape {pred_tensor.shape}")
            
            # Update progress tracking
            batch_size = images.size(0)
            processed_samples += batch_size
            running_loss += losses['total_loss'].item() * batch_size
            
            # Update progress bar (less frequently to reduce overhead)
            if batch_idx % 5 == 0:
                progress_bar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
            
            # Visualize training predictions 5 times per epoch
            if batch_idx % train_viz_interval == 0:
                try:
                    # Ensure predictions is a valid tensor
                    if isinstance(predictions, torch.Tensor) and predictions.dim() == 4:
                        viz_name = f'train_epoch{self.epoch+1}_batch{batch_idx}'
                        self._visualize_predictions(images.detach(), predictions.detach(), targets.detach(), prefix=viz_name)
                        print(f"  Visualized training batch {batch_idx}")
                except Exception as e:
                    print(f"Warning: Failed to visualize training predictions: {e}")
            
            self.train_step += 1
            
            # Clear cache periodically to prevent memory fragmentation
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Update scheduler - Only for non-batch-based schedulers
        if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR) and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        # Compute epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_loss = running_loss / processed_samples
        
        # Get final metrics for the epoch
        metrics = self.train_metrics.get_metrics()
        
        # Log epoch summary
        print(f"\nEpoch {self.epoch+1} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Mean IoU: {metrics.get('mean_iou', 0):.4f}")
        
        self.epoch += 1
        return metrics
    
    def _plot_learning_rate_curve(self):
        """Plot learning rate curve and save to disk"""
        plt.figure(figsize=(10, 5))
        
        # Get learning rates for each component
        component_names = ['camera', 'lidar', 'fusion', 'head']
        for i, name in enumerate(component_names):
            if i < len(self.optimizer.param_groups):
                lr_history = self.metrics_logger.get_lr_history(i)
                if lr_history:
                    plt.plot(lr_history, label=name)
        
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        lr_curve_path = self.viz_dir / 'learning_rate_curve.png'
        plt.savefig(lr_curve_path)
        plt.close()
        
        # Add to TensorBoard
        try:
            img = plt.imread(lr_curve_path)
            self.writer.add_image('LearningRate/Curve', img.transpose(2, 0, 1), self.epoch)
        except Exception as e:
            print(f"Could not add learning rate curve to TensorBoard: {e}")
    
    def _visualize_predictions(self, images, predictions, targets, prefix='train'):
        """Visualize model predictions and save to disk."""
        try:
            import matplotlib.pyplot as plt
            
            # Ensure the visualization directory exists
            if not hasattr(self, 'viz_dir'):
                self.viz_dir = self.output_dir / 'visualizations'
                self.viz_dir.mkdir(parents=True, exist_ok=True)
                
            # Reduce the number of samples to visualize
            num_samples = min(2, images.size(0))  # Only visualize 2 samples max
            
            # Use a smaller figure size
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
            
            # If only one sample, make sure axes is 2D
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            # Get class names
            class_names = ['background', 'drivable_area', 'lane_divider', 
                          'road_divider', 'walkway', 'ped_crossing']
            
            # Get class colors from config
            class_colors = self.config.get('class_colors', [
                [252, 252, 252],  # Background
                [166, 206, 227],  # Drivable Area
                [202, 178, 214],  # Road Divider
                [106, 61, 154],   # Lane Divider
                [224, 74, 76],    # Walkway
                [251, 154, 153]   # Pedestrian Crossing
            ])
            
            for i in range(num_samples):
                # Original image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
                axes[i, 0].imshow(img)
                axes[i, 0].set_title('Input Image')
                axes[i, 0].axis('off')
                
                # Ground truth segmentation
                if targets.dim() == 4:  # [B, C, H, W]
                    target = targets[i].argmax(dim=0).cpu().numpy()
                else:
                    target = targets[i].cpu().numpy()
                    
                target_rgb = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
                for cls_idx, color in enumerate(class_colors):
                    if cls_idx < len(class_colors):  # Safety check
                        target_rgb[target == cls_idx] = color
                    
                axes[i, 1].imshow(target_rgb)
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                # Predicted segmentation
                pred = predictions[i].argmax(dim=0).cpu().numpy()
                pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                for cls_idx, color in enumerate(class_colors):
                    if cls_idx < len(class_colors):  # Safety check
                        pred_rgb[pred == cls_idx] = color
                    
                axes[i, 2].imshow(pred_rgb)
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
            
            # Add a color legend at the bottom
            legend_elements = [plt.Rectangle((0, 0), 1, 1, color=[c/255 for c in color], label=name) 
                              for color, name in zip(class_colors, class_names)]
            fig.legend(handles=legend_elements, loc='lower center', ncol=len(class_names), 
                      bbox_to_anchor=(0.5, 0), fontsize=10)
            
            # Adjust layout to make room for the legend
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            
            # Save figure
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            viz_path = self.viz_dir / f'{prefix}_predictions_{self.epoch}_{timestamp}.png'
            plt.savefig(viz_path)
            plt.close(fig)
            
            print(f"  Saved visualization to {viz_path}")
            return True
        except Exception as e:
            print(f"Warning: Visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate(self):
        """Run validation on the validation dataset"""
        # Set models to evaluation mode
        self.camera_backbone.eval()
        self.lidar_backbone.eval()
        self.fusion.eval()
        self.head.eval()
        self.image_projections.eval()
        self.lidar_projections.eval()
        
        # Reset validation metrics
        self.val_metrics.reset()
        
        # Initialize tracking variables
        running_loss = 0.0
        processed_samples = 0
        val_start_time = time.time()
        
        # Create progress bar
        progress_bar = tqdm(self.val_loader, desc=f"Validating")
        
        # Calculate visualization interval - show 3 visualizations per validation run
        val_viz_interval = max(1, len(self.val_loader) // 3)
        print(f"Will visualize validation predictions every {val_viz_interval} batches (3 times per validation)")
        
        # Validate
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                images = batch.get('image', None)
                lidar_points = batch.get('lidar', None)
                targets = batch.get('bev_label', None)
                calibs = batch.get('calib', None)
                
                # Skip batch if missing required data
                if images is None or lidar_points is None or targets is None or calibs is None:
                    print(f"Warning: Skipping batch {batch_idx} due to missing data")
                    continue
                
                # Check tensor dimensions to prevent errors
                try:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Ensure targets are in the correct format
                    # If targets are [B, 1, H, W] with class indices, convert to [B, H, W]
                    if targets.dim() == 4 and targets.size(1) == 1:
                        targets = targets.squeeze(1)
                    
                    # Process lidar points
                    processed_lidar = []
                    for points in lidar_points:
                        processed_lidar.append(points.to(self.device))
                    
                    # Extract features
                    image_features_dict = self.camera_backbone(images)
                    
                    # Process each point cloud and get features
                    # The lidar_backbone only takes one argument (points)
                    lidar_features = {}
                    for i, points in enumerate(processed_lidar):
                        # Process each point cloud individually
                        features = self.lidar_backbone(points.unsqueeze(0))
                        # Store features for each batch item
                        if i == 0:
                            # Initialize dictionaries with the right structure
                            for key in features['bev_features']:
                                lidar_features[key] = []
                        
                        # Collect features from each batch item
                        for key in features['bev_features']:
                            lidar_features[key].append(features['bev_features'][key])
                    
                    # Create dictionary of lidar features for each stage with projections
                    lidar_features_dict = {}
                    for stage in ['stage1', 'stage2', 'stage3']:
                        # Stack features if they have the same shape
                        if all(feat.shape[2:] == lidar_features[stage][0].shape[2:] for feat in lidar_features[stage]):
                            stacked_features = torch.cat(lidar_features[stage], dim=0)
                            lidar_features_dict[stage] = self.lidar_projections[stage](stacked_features)
                        else:
                            # Fallback: just use the first batch item's features
                            lidar_features_dict[stage] = self.lidar_projections[stage](lidar_features[stage][0])
                    
                    # Fuse features
                    fused_feats, _ = self.fusion(lidar_features_dict, image_features_dict)
                    
                    # Get predictions and compute loss
                    with amp.autocast():
                        # Convert targets to one-hot encoding for loss calculation
                        if targets.dim() == 3:  # [B, H, W]
                            num_classes = self.config['num_classes']
                            targets_one_hot = torch.zeros(
                                targets.size(0), num_classes, targets.size(1), targets.size(2),
                                device=self.device
                            )
                            for cls in range(num_classes):
                                targets_one_hot[:, cls] = (targets == cls).float()
                            
                            # Compute losses with one-hot encoded targets
                            try:
                                losses = self.head(fused_feats, targets_one_hot)
                                if not isinstance(losses, dict):
                                    # If head returns predictions instead of losses, create a dummy loss
                                    print(f"Warning: head returned predictions instead of losses in batch {batch_idx}")
                                    losses = {'total_loss': torch.tensor(0.0, device=self.device)}
                            except Exception as e:
                                print(f"Error computing losses in batch {batch_idx}: {str(e)}")
                                # Create dummy losses to continue validation
                                losses = {'total_loss': torch.tensor(0.0, device=self.device)}
                        else:
                            # Targets are already one-hot encoded
                            try:
                                losses = self.head(fused_feats, targets)
                                if not isinstance(losses, dict):
                                    # If head returns predictions instead of losses, create a dummy loss
                                    print(f"Warning: head returned predictions instead of losses in batch {batch_idx}")
                                    losses = {'total_loss': torch.tensor(0.0, device=self.device)}
                            except Exception as e:
                                print(f"Error computing losses in batch {batch_idx}: {str(e)}")
                                # Create dummy losses to continue validation
                                losses = {'total_loss': torch.tensor(0.0, device=self.device)}
                        
                        # Get predictions without targets (for metrics)
                        predictions = self.head(fused_feats)  # Only pass fused_feats, not targets
                        
                        # Ensure targets have the correct shape for metrics update
                        # The predictions are [B, C, H, W] and targets might be [B, 1, H, W] or [B, H, W]
                        if targets.dim() == 4 and targets.shape[1] == 1:
                            # Convert [B, 1, H, W] to [B, H, W]
                            targets = targets.squeeze(1)
                        
                        # Handle shape mismatch - convert class indices to appropriate format for metrics
                        # Check if we're getting a shape mismatch warning
                        if predictions.shape[1] != targets.shape[1] if targets.dim() > 3 else predictions.shape[1] != 1:
                            # If targets are class indices [B, H, W], pass the full predictions tensor
                            if targets.dim() == 3:
                                # Ensure predictions is a 4D tensor with shape [B, C, H, W]
                                if isinstance(predictions, dict):
                                    # If predictions is a dictionary, extract logits
                                    pred_tensor = predictions['logits'] if 'logits' in predictions else predictions
                                else:
                                    # If predictions is already a tensor
                                    pred_tensor = predictions
                                
                                # Verify we have a 4D tensor before passing to metrics
                                if pred_tensor.dim() == 4:
                                    self.val_metrics.update(pred_tensor, targets)
                                else:
                                    print(f"Warning: Expected 4D predictions tensor, got shape {pred_tensor.shape}")
                            else:
                                print(f"Warning: Incompatible target shape: {targets.shape} vs prediction shape: {predictions.shape}")
                        else:
                            # If shapes match (both one-hot), update directly
                            # Ensure predictions is a 4D tensor
                            if isinstance(predictions, dict):
                                pred_tensor = predictions['logits'] if 'logits' in predictions else predictions
                            else:
                                pred_tensor = predictions
                            
                            if pred_tensor.dim() == 4:
                                self.val_metrics.update(pred_tensor, targets)
                            else:
                                print(f"Warning: Expected 4D predictions tensor, got shape {pred_tensor.shape}")
                        
                        #print("  SegmentationMetrics.update completed")
                    
                    # Update progress tracking
                    batch_size = images.size(0)
                    processed_samples += batch_size
                    running_loss += losses['total_loss'].item() * batch_size
                    
                    # Update progress bar (less frequently to reduce overhead)
                    if batch_idx % 10 == 0:
                        try:
                            progress_bar.set_postfix({
                                'loss': f"{losses['total_loss'].item():.4f}"
                            })
                        except Exception as e:
                            # Ignore errors in progress bar updates
                            print(f"Warning: Error updating progress bar: {str(e)}")
                    
                    # Visualize validation predictions 3 times per validation run
                    if batch_idx % val_viz_interval == 0:
                        try:
                            # Ensure predictions is a valid tensor
                            if isinstance(predictions, torch.Tensor) and predictions.dim() == 4:
                                viz_name = f'val_epoch{self.epoch+1}_batch{batch_idx}'
                                self._visualize_predictions(images.detach(), predictions.detach(), targets.detach(), prefix=viz_name)
                                print(f"  Visualized validation batch {batch_idx}")
                        except Exception as e:
                            print(f"Warning: Failed to visualize validation predictions: {e}")
                
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    # Continue with next batch instead of crashing
                    continue
                
                # Clear cache periodically to prevent memory fragmentation
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
        
        # Skip metrics calculation if no samples were processed
        if processed_samples == 0:
            print("Warning: No samples were successfully processed during validation")
            return {'loss': float('inf'), 'mean_iou': 0.0}
        
        # Compute final metrics
        metrics = self.val_metrics.get_metrics()
        avg_loss = running_loss / processed_samples
        metrics['loss'] = avg_loss
        
        # Compute validation time
        val_time = time.time() - val_start_time
        
        # Log validation summary
        print(f"\nValidation Summary:")
        print(f"  Time: {val_time:.2f}s")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Mean IoU: {metrics.get('mean_iou', 0):.4f}")
        
        return metrics
    
    def _plot_metrics_history(self):
        """Plot metrics history and save to disk"""
        # This method is no longer needed
        pass
    
    def _check_early_stopping(self, current_score):
        """Check if training should be stopped early"""
        # This method is no longer needed
        pass
    
    def _plot_confusion_matrix(self, metrics):
        """Plot confusion matrix and save to disk"""
        # This method is no longer needed
        pass
    
    def save_checkpoint(self, path: str = None, is_best: bool = False):
        """Save model checkpoint"""
        if path is None:
            path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': self.epoch,
            'train_step': self.train_step,
            'best_val_iou': self.best_val_iou,
            'camera_backbone': self.camera_backbone.state_dict(),
            'lidar_backbone': self.lidar_backbone.state_dict(),
            'fusion': self.fusion.state_dict(),
            'head': self.head.state_dict(),
            'image_projections': self.image_projections.state_dict(),
            'lidar_projections': self.lidar_projections.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'config': self.config
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
        # If this is a best checkpoint, save a copy
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        print(f"Loading checkpoint from {path}")
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model weights
            self.camera_backbone.load_state_dict(checkpoint['camera_backbone'])
            self.lidar_backbone.load_state_dict(checkpoint['lidar_backbone'])
            self.fusion.load_state_dict(checkpoint['fusion'])
            self.head.load_state_dict(checkpoint['head'])
            
            # Load projection layers if available
            if 'image_projections' in checkpoint:
                self.image_projections.load_state_dict(checkpoint['image_projections'])
            if 'lidar_projections' in checkpoint:
                self.lidar_projections.load_state_dict(checkpoint['lidar_projections'])
            
            # Load optimizer and scheduler
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            
            # Load training state
            self.epoch = checkpoint['epoch']
            self.train_step = checkpoint['train_step']
            self.best_val_iou = checkpoint['best_val_iou']
            
            print(f"Successfully loaded checkpoint from epoch {self.epoch}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def export_model(self, format='torchscript', quantize=False):
        """Export model to different formats for deployment"""
        # This method is no longer needed
        pass
    
    def profile_model(self, batch_size=1):
        """Profile model performance and memory usage"""
        # This method is no longer needed
        pass
    
    def log_metrics(self, metrics: Dict, prefix: str = ''):
        """Log metrics to console and TensorBoard"""
        # This method is no longer needed
        pass


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized point clouds."""
    # Unpack batch
    images = []
    lidar_points = []
    bev_labels = []
    calibs = []
    sample_tokens = []
    scene_tokens = []
    
    for item in batch:
        if 'image' in item:
            images.append(item['image'])
        if 'lidar' in item:
            lidar_points.append(item['lidar'])
        if 'bev_label' in item:
            # Standardize dimensions: ensure labels are 3D tensors [C, H, W]
            label = item['bev_label']
            if label.dim() == 2:  # If [H, W], convert to [1, H, W]
                label = label.unsqueeze(0)
            elif label.dim() == 3 and label.size(0) != 1:  # If [H, W, C], permute to [C, H, W]
                label = label.permute(2, 0, 1)
            bev_labels.append(label)
        if 'calib' in item:
            calibs.append(item['calib'])
        if 'sample_token' in item:
            sample_tokens.append(item['sample_token'])
        if 'scene_token' in item:
            scene_tokens.append(item['scene_token'])
    
    # Create batch dictionary
    batch_dict = {}
    
    # Stack tensors where possible
    if images:
        batch_dict['image'] = torch.stack(images)
    
    if bev_labels:
        batch_dict['bev_label'] = torch.stack(bev_labels)
    
    # Handle lidar points (can't stack due to variable sizes)
    if lidar_points:
        batch_dict['lidar'] = lidar_points
    
    # Add calibration data
    if calibs:
        batch_dict['calib'] = calibs
    
    # Add tokens if available
    if sample_tokens:
        batch_dict['sample_token'] = sample_tokens
    if scene_tokens:
        batch_dict['scene_token'] = scene_tokens
    
    return batch_dict


def main():
    parser = argparse.ArgumentParser(description='Train BEV Fusion model')
    parser.add_argument('--model-config', type=str, required=True, help='Path to model configuration YAML')
    parser.add_argument('--train-config', type=str, required=True, help='Path to training configuration YAML')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to NuScenes dataset')
    parser.add_argument('--bev-labels-dir', type=str, required=True, help='Path to precomputed BEV labels')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with smaller dataset')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--num-workers', type=int, default=None, help='Override number of workers from config')
    parser.add_argument('--optimize-memory', action='store_true', help='Enable memory optimization')
    parser.add_argument('--image-size', type=str, default='672,376', help='Image size to use (width,height)')
    parser.add_argument('--max-lidar-points', type=int, default=20000, help='Maximum number of lidar points to use')
    parser.add_argument('--simulate-vlp16', action='store_true', help='Simulate VLP-16 Puck Hi-Res LiDAR')
    parser.add_argument('--early-stopping', action='store_true', help='Use early stopping to prevent overfitting')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for regularization')
    parser.add_argument('--augmentation-strength', type=str, default='medium', 
                      choices=['none', 'light', 'medium', 'strong'], 
                      help='Strength of data augmentation')
    args = parser.parse_args()
    
    # Load configurations
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.train_config, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save configurations for reproducibility
    shutil.copy(args.model_config, output_dir / os.path.basename(args.model_config))
    shutil.copy(args.train_config, output_dir / os.path.basename(args.train_config))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Override batch size and num_workers if specified
    if args.batch_size is not None:
        train_config['train']['batch_size'] = args.batch_size
        train_config['val']['batch_size'] = args.batch_size
        print(f"Overriding batch size to {args.batch_size}")
    
    if args.num_workers is not None:
        train_config['train']['num_workers'] = args.num_workers
        print(f"Overriding num_workers to {args.num_workers}")
    
    # Parse image size
    try:
        width, height = map(int, args.image_size.split(','))
        print(f"Using image size: {width}x{height}")
        # Update model config with new image size
        model_config['input_config']['image_size'] = [height, width]
    except:
        print(f"Invalid image size format: {args.image_size}, using default")
        height, width = model_config['input_config']['image_size']
    
    # Update max lidar points
    max_lidar_points = args.max_lidar_points
    if max_lidar_points:
        print(f"Using max lidar points: {max_lidar_points}")
        model_config['input_config']['max_points_per_frame'] = max_lidar_points
        model_config['lidar_backbone']['max_voxels'] = min(max_lidar_points, model_config['lidar_backbone']['max_voxels'])
    
    # Debug settings
    debug_mode = args.debug
    if debug_mode:
        print("Debug mode enabled - using smaller dataset")
        subset_size = 9667 # Small subset for debugging
    
    # Memory optimization
    if args.optimize_memory:
        print("Enabling memory optimization")
        # Reduce precision for point cloud range and voxel size
        model_config['lidar_backbone']['point_cloud_range'] = [float(x) for x in model_config['lidar_backbone']['point_cloud_range']]
        model_config['lidar_backbone']['voxel_size'] = [float(x) for x in model_config['lidar_backbone']['voxel_size']]
        
        # Enable memory efficient attention if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmark mode for better performance
    torch.backends.cudnn.benchmark = True
    
    # Create combined config for model
    config = {
        'num_classes': model_config['segmentation_head']['num_classes'],
        'voxel_size': model_config['lidar_backbone']['voxel_size'],
        'point_cloud_range': model_config['lidar_backbone']['point_cloud_range'],
        'pretrained': model_config['camera_backbone']['pretrained'],
        'dropout': model_config['segmentation_head']['dropout'],
        'use_focal_loss': model_config['segmentation_head']['use_focal_loss'],
        'lr_camera': train_config['optimizer']['lr_camera'],
        'lr_lidar': train_config['optimizer']['lr_lidar'],
        'lr_fusion': train_config['optimizer']['lr_fusion'],
        'lr_head': train_config['optimizer']['lr_head'],
        'lr_min': train_config['scheduler']['eta_min'],
        'weight_decay': train_config['optimizer']['weight_decay'],
        'epochs': train_config['train']['epochs'],
        'class_colors': [
            [252, 252, 252],    # background - #fcfcfc
            [166, 206, 227],    # drivable area - #a6cee3
            [202, 178, 214],    # road divider - #cab2d6
            [106, 61, 154],     # lane divider - #6a3d9a
            [224, 74, 76],      # walkway - #e04a4c
            [251, 154, 153]     # pedestrian crossing - #fb9a99
        ]
    }
    
    # Resume training if specified
    if args.resume:
        config['resume_from'] = args.resume
    
    # Create image transform for resizing
    def image_transform(image):
        # Resize image to target size
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        # Convert to tensor and normalize
        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1) / 255.0
        return tensor_image
    
    # Create datasets
    print("Creating datasets...")
    
    # Print VLP-16 simulation status
    if args.simulate_vlp16:
        print("Simulating VLP-16 Puck Hi-Res LiDAR (-10° to 10° FOV, 16 channels)")
    
    # Configure augmentation based on selected strength
    augmentation_config = {
        'none': None,
        'light': BEVAugmentor(
            flip_prob=0.3,
            rotate_prob=0.3,
            rotate_range=(-5, 5),
            scale_prob=0.0,
            translate_prob=0.0,
            cutout_prob=0.0,
            mixup_prob=0.0,
            feature_noise_prob=0.3
        ),
        'medium': BEVAugmentor(
            flip_prob=0.5,
            rotate_prob=0.5,
            rotate_range=(-10, 10),
            scale_prob=0.5,
            translate_prob=0.5,
            cutout_prob=0.3,
            mixup_prob=0.2,
            feature_noise_prob=0.5
        ),
        'strong': BEVAugmentor(
            flip_prob=0.7,
            rotate_prob=0.7,
            rotate_range=(-15, 15),
            scale_prob=0.7,
            translate_prob=0.7,
            cutout_prob=0.5,
            cutout_max_boxes=3,
            mixup_prob=0.4,
            feature_noise_prob=0.7,
            color_jitter_prob=0.7
        )
    }
    
    # Create augmentor based on selected strength
    augmentor = augmentation_config[args.augmentation_strength]
    if augmentor:
        print(f"Using {args.augmentation_strength} data augmentation")
    
    # Train dataset
    base_train_dataset = PrecomputedBEVDataset(
        dataroot=args.dataroot,
        bev_labels_dir=args.bev_labels_dir,
        split='train',
        return_tokens=True,
        transform=image_transform,
        simulate_vlp16=args.simulate_vlp16
    )
    
    # Wrap with augmentation if enabled
    if augmentor:
        train_dataset = RegularizedTrainingWrapper(
            base_train_dataset,
            augmentor=augmentor,
            dropout_rate=args.dropout,
            weight_decay=args.weight_decay
        )
    else:
        train_dataset = base_train_dataset
    
    # Validation dataset
    val_dataset = PrecomputedBEVDataset(
        dataroot=args.dataroot,
        bev_labels_dir=args.bev_labels_dir,
        split='val',
        return_tokens=True,
        transform=image_transform,
        simulate_vlp16=args.simulate_vlp16
    )
    
    # Limit dataset size in debug mode
    if debug_mode:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(subset_size, len(val_dataset))))
    
    print(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['train']['batch_size'],
        shuffle=True,
        num_workers=train_config['train']['num_workers'],
        pin_memory=train_config['train']['pin_memory'],
        collate_fn=custom_collate_fn,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['val']['batch_size'],
        shuffle=False,
        num_workers=train_config['train']['num_workers'],
        pin_memory=train_config['train']['pin_memory'],
        collate_fn=custom_collate_fn,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        config, 
        train_config,
        train_loader, 
        val_loader, 
        device, 
        output_dir,
        use_early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout
    )
    
    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Batch size: {train_config['train']['batch_size']}")
    print(f"Number of workers: {train_config['train']['num_workers']}")
    print(f"Weight decay: {args.weight_decay}, Dropout: {args.dropout}")
    if args.early_stopping:
        print(f"Early stopping enabled with patience: {args.patience}")
    
    start_time = time.time()
    
    try:
        for epoch in range(trainer.epoch, config['epochs']):
            # Train
            train_metrics = trainer.train_epoch()
            
            # Initialize val_metrics with default values
            val_metrics = {'mean_iou': 0.0}
            
            # Validate
            if (epoch + 1) % train_config['val']['interval'] == 0:
                val_metrics = trainer.validate()
                
                # Early stopping check if enabled
                if args.early_stopping:
                    should_stop = trainer.early_stopping.track_metrics(
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        model=trainer.model,
                        epoch=epoch,
                        optimizer=trainer.optimizer,
                        scheduler=trainer.scheduler
                    )
                    
                    if should_stop:
                        print(f"Early stopping triggered at epoch {epoch+1}.")
                        break
                
                # Update learning rate if using ReduceLROnPlateau
                if isinstance(trainer.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    trainer.scheduler.step(val_metrics.get('mean_iou', 0))
            
            # Save checkpoint if best validation IoU
            if val_metrics.get('mean_iou', 0) > trainer.best_val_iou:
                trainer.best_val_iou = val_metrics['mean_iou']
                trainer.save_checkpoint(is_best=True)
                print(f"Saved best model with IoU: {trainer.best_val_iou:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % train_config['checkpoint']['save_interval'] == 0:
                trainer.save_checkpoint()
                
            # Update learning rate for other schedulers
            if trainer.scheduler is not None and not isinstance(trainer.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                trainer.scheduler.step()
                
            # Clear cache between epochs
            torch.cuda.empty_cache()
            
            # Print memory stats
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB allocated, "
                      f"{torch.cuda.memory_reserved(0) / 1024**2:.2f}MB reserved")
        
        # Training completed
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s!")
        print(f"Best validation IoU: {trainer.best_val_iou:.4f}")
        
        # Print early stopping summary if used
        if args.early_stopping:
            summary = trainer.early_stopping.get_summary()
            print(f"Early stopping summary:")
            print(f"- Best epoch: {summary['best_epoch']}")
            print(f"- Best score: {summary['best_score']:.4f}")
            print(f"- Stopped early: {summary['stopped_early']}")
            
        print("\nTraining pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save checkpoint on interrupt
        trainer.save_checkpoint(path=output_dir / 'checkpoints' / 'interrupt_checkpoint.pth')
        print("Saved interrupt checkpoint.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()