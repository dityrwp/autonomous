import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.backbones import EfficientNetV2Backbone, SECONDBackbone
from models.fusion import BEVFusion
from models.heads import BEVSegmentationHead
from utils.metrics import SegmentationMetrics, MetricsLogger


class Trainer:
    def __init__(
        self,
        config: Dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize model components
        self.camera_backbone = EfficientNetV2Backbone(
            pretrained=config['pretrained']
        ).to(device)
        
        self.lidar_backbone = SECONDBackbone(
            voxel_size=config['voxel_size'],
            point_cloud_range=config['point_cloud_range']
        ).to(device)
        
        self.fusion = BEVFusion(
            lidar_channels=128,
            image_channels={
                'stage3': 256,
                'stage4': 384,
                'stage5': 512
            },
            output_channels=128,
            bev_height=128,
            bev_width=128,
            voxel_size=config['voxel_size']
        ).to(device)
        
        self.head = BEVSegmentationHead(
            in_channels=128,
            hidden_channels=128,
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            use_focal_loss=config['use_focal_loss']
        ).to(device)
        
        # Initialize metrics
        self.train_metrics = SegmentationMetrics(config['num_classes'], device)
        self.val_metrics = SegmentationMetrics(config['num_classes'], device)
        self.metrics_logger = MetricsLogger()
        
        # Optimization
        self.optimizer = optim.AdamW([
            {'params': self.camera_backbone.parameters(), 'lr': config['lr_camera']},
            {'params': self.lidar_backbone.parameters(), 'lr': config['lr_lidar']},
            {'params': self.fusion.parameters(), 'lr': config['lr_fusion']},
            {'params': self.head.parameters(), 'lr': config['lr_head']}
        ], weight_decay=config['weight_decay'])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['lr_min']
        )
        
        # AMP scaler
        self.scaler = amp.GradScaler()
        
        # Metrics tracking
        self.best_val_iou = 0.0
        self.train_step = 0
        self.epoch = 0
        
        # Load checkpoint if specified
        if config.get('resume_from'):
            self.load_checkpoint(config['resume_from'])
    
    def train_epoch(self):
        self.camera_backbone.train()
        self.lidar_backbone.train()
        self.fusion.train()
        self.head.train()
        self.train_metrics.reset()
        
        for batch in self.train_loader:
            # Move data to device
            images = batch['images'].to(self.device)
            points = batch['points'].to(self.device)
            calib = {k: v.to(self.device) for k, v in batch['calib'].items()}
            targets = batch['targets'].to(self.device)
            
            # Forward pass with AMP
            with amp.autocast():
                # Extract features
                image_feats = self.camera_backbone(images)
                lidar_feats = self.lidar_backbone(points)['bev_features']['stage3']
                
                # Fuse features
                fused_feats = self.fusion(
                    lidar_feats,
                    image_feats,
                    calib
                )
                
                # Predict and compute loss
                losses = self.head(fused_feats, targets)
                
                # Get predictions for metrics
                predictions = self.head(fused_feats)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            self.metrics_logger.update_loss(losses['total_loss'].item())
            self.train_metrics.update(predictions.detach(), targets)
            
            # Log metrics
            if self.train_step % self.config['log_interval'] == 0:
                metrics = self.train_metrics.get_metrics()
                self.metrics_logger.update_metrics(metrics)
                self.log_metrics({**losses, **metrics}, prefix='train')
            
            self.train_step += 1
        
        # Update scheduler
        self.scheduler.step()
        self.epoch += 1
        
        # Get final metrics for the epoch
        metrics = self.train_metrics.get_metrics()
        self.metrics_logger.update_metrics(metrics)
        return metrics
    
    @torch.no_grad()
    def validate(self):
        if not self.val_loader:
            return {}
        
        self.camera_backbone.eval()
        self.lidar_backbone.eval()
        self.fusion.eval()
        self.head.eval()
        self.val_metrics.reset()
        
        val_losses = []
        
        for batch in self.val_loader:
            # Move data to device
            images = batch['images'].to(self.device)
            points = batch['points'].to(self.device)
            calib = {k: v.to(self.device) for k, v in batch['calib'].items()}
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            image_feats = self.camera_backbone(images)
            lidar_feats = self.lidar_backbone(points)['bev_features']['stage3']
            fused_feats = self.fusion(lidar_feats, image_feats, calib)
            
            # Get predictions and loss
            predictions = self.head(fused_feats)
            losses = self.head(fused_feats, targets)
            
            # Update metrics
            self.val_metrics.update(predictions, targets)
            val_losses.append(losses['total_loss'].item())
        
        # Compute final metrics
        metrics = self.val_metrics.get_metrics()
        metrics['loss'] = sum(val_losses) / len(val_losses)
        
        # Log validation metrics
        self.log_metrics(metrics, prefix='val')
        
        return metrics
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'epoch': self.epoch,
            'train_step': self.train_step,
            'best_val_iou': self.best_val_iou,
            'camera_backbone': self.camera_backbone.state_dict(),
            'lidar_backbone': self.lidar_backbone.state_dict(),
            'fusion': self.fusion.state_dict(),
            'head': self.head.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'metrics_history': self.metrics_logger.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.train_step = checkpoint['train_step']
        self.best_val_iou = checkpoint['best_val_iou']
        self.camera_backbone.load_state_dict(checkpoint['camera_backbone'])
        self.lidar_backbone.load_state_dict(checkpoint['lidar_backbone'])
        self.fusion.load_state_dict(checkpoint['fusion'])
        self.head.load_state_dict(checkpoint['head'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        if 'metrics_history' in checkpoint:
            self.metrics_logger.history = checkpoint['metrics_history']
    
    def log_metrics(self, metrics: Dict, prefix: str = ''):
        """Log metrics (implement your preferred logging here)"""
        # Example: Print to console
        print(f"\n{prefix.upper()} Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


def main():
    # Load your configuration
    config = {
        'pretrained': True,
        'voxel_size': [0.8, 0.8, 0.8],
        'point_cloud_range': [-51.2, -51.2, -5, 51.2, 51.2, 3],
        'num_classes': 3,
        'dropout': 0.1,
        'use_focal_loss': True,
        'lr_camera': 1e-4,
        'lr_lidar': 1e-4,
        'lr_fusion': 1e-4,
        'lr_head': 1e-4,
        'lr_min': 1e-6,
        'weight_decay': 0.01,
        'epochs': 100,
        'log_interval': 10
    }
    
    # Create your data loaders
    train_loader = None  # Implement your data loading
    val_loader = None    # Implement your data loading
    
    # Initialize trainer
    trainer = Trainer(config, train_loader, val_loader)
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_metrics = trainer.train_epoch()
        print("\nTraining metrics:")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Validate
        val_metrics = trainer.validate()
        if val_metrics:
            print("\nValidation metrics:")
            for k, v in val_metrics.items():
                print(f"{k}: {v:.4f}")
        
        # Save checkpoint if best validation IoU
        if val_metrics.get('mean_iou', 0) > trainer.best_val_iou:
            trainer.best_val_iou = val_metrics['mean_iou']
            trainer.save_checkpoint('best_model.pth')
        
        # Regular checkpoint
        if epoch % 10 == 0:
            trainer.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
        # End epoch in metrics logger
        trainer.metrics_logger.epoch_end()


if __name__ == '__main__':
    main()