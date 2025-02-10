import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2

from models.backbones import EfficientNetV2Backbone, SECONDBackbone
from models.fusion import BEVFusion
from models.heads import BEVSegmentationHead
from utils.metrics import SegmentationMetrics


class Tester:
    def __init__(
        self,
        config: Dict,
        checkpoint_path: str,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = device
        
        # Initialize model components
        self.camera_backbone = EfficientNetV2Backbone(
            pretrained=False  # No need for pretrained during testing
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
            dropout=0.0  # No dropout during testing
        ).to(device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(config['num_classes'], device)
        
        # Set all models to eval mode
        self.camera_backbone.eval()
        self.lidar_backbone.eval()
        self.fusion.eval()
        self.head.eval()
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.camera_backbone.load_state_dict(checkpoint['camera_backbone'])
        self.lidar_backbone.load_state_dict(checkpoint['lidar_backbone'])
        self.fusion.load_state_dict(checkpoint['fusion'])
        self.head.load_state_dict(checkpoint['head'])
    
    @torch.no_grad()
    def test_sequence(
        self,
        test_loader: DataLoader,
        save_predictions: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Test on a sequence of data
        Args:
            test_loader: DataLoader for test data
            save_predictions: Whether to save prediction visualizations
            output_dir: Directory to save predictions (if save_predictions is True)
        Returns:
            Dictionary of metrics
        """
        self.metrics.reset()
        inference_times = []
        
        if save_predictions and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, batch in enumerate(test_loader):
            # Move data to device
            images = batch['images'].to(self.device)
            points = batch['points'].to(self.device)
            calib = {k: v.to(self.device) for k, v in batch['calib'].items()}
            
            # Time the inference
            start_time = time.time()
            
            # Forward pass
            image_feats = self.camera_backbone(images)
            lidar_feats = self.lidar_backbone(points)['bev_features']['stage3']
            fused_feats = self.fusion(lidar_feats, image_feats, calib)
            predictions = self.head(fused_feats)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Update metrics if targets are available
            if 'targets' in batch:
                targets = batch['targets'].to(self.device)
                self.metrics.update(predictions, targets)
            
            # Save predictions if requested
            if save_predictions and output_dir:
                self.save_prediction_visualization(
                    predictions[0],  # Take first batch item
                    os.path.join(output_dir, f'pred_{i:06d}.png')
                )
        
        # Compute metrics
        metrics = self.metrics.get_metrics() if 'targets' in batch else {}
        
        # Add timing metrics
        metrics['mean_inference_time'] = np.mean(inference_times)
        metrics['std_inference_time'] = np.std(inference_times)
        metrics['fps'] = 1.0 / np.mean(inference_times)
        
        return metrics
    
    def save_prediction_visualization(
        self,
        prediction: torch.Tensor,
        save_path: str,
        threshold: float = 0.5
    ):
        """
        Save a visualization of the prediction
        Args:
            prediction: [C, H, W] prediction tensor
            save_path: Path to save the visualization
            threshold: Classification threshold
        """
        # Convert prediction to numpy
        pred = prediction.cpu().numpy()
        
        # Create RGB visualization
        vis = np.zeros((pred.shape[1], pred.shape[2], 3), dtype=np.uint8)
        
        # Color code for each class
        colors = [
            (255, 0, 0),    # Red for class 0
            (0, 255, 0),    # Green for class 1
            (0, 0, 255)     # Blue for class 2
        ]
        
        # Add each class prediction with its color
        for i in range(pred.shape[0]):
            mask = pred[i] > threshold
            vis[mask] = colors[i]
        
        # Save visualization
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    @torch.no_grad()
    def inference_single(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        calib: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Run inference on a single sample
        Args:
            images: [1, C, H, W] image tensor
            points: [1, N, 4] point cloud tensor
            calib: Calibration matrices
        Returns:
            [1, num_classes, H, W] prediction tensor
        """
        # Move inputs to device
        images = images.to(self.device)
        points = points.to(self.device)
        calib = {k: v.to(self.device) for k, v in calib.items()}
        
        # Forward pass
        image_feats = self.camera_backbone(images)
        lidar_feats = self.lidar_backbone(points)['bev_features']['stage3']
        fused_feats = self.fusion(lidar_feats, image_feats, calib)
        predictions = self.head(fused_feats)
        
        return predictions


def main():
    # Load your configuration
    config = {
        'voxel_size': [0.8, 0.8, 0.8],
        'point_cloud_range': [-51.2, -51.2, -5, 51.2, 51.2, 3],
        'num_classes': 3
    }
    
    # Initialize tester
    tester = Tester(
        config=config,
        checkpoint_path='best_model.pth'
    )
    
    # Create test data loader
    test_loader = None  # Implement your test data loading
    
    # Run testing
    if test_loader is not None:
        metrics = tester.test_sequence(
            test_loader=test_loader,
            save_predictions=True,
            output_dir='predictions'
        )
        
        # Print metrics
        print("\nTest Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    
    else:
        print("Please implement test data loading to proceed with testing.")


if __name__ == '__main__':
    main()
