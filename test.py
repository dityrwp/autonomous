import os
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Union, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.backbones import EfficientNetV2Backbone, SECONDBackbone
from models.fusion import BEVFusion
from models.heads import BEVSegmentationHead
from datasets.precomputed_bev_dataset import PrecomputedBEVDataset
from utils.metrics import SegmentationMetrics


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized LiDAR data."""
    elem = batch[0]
    result = {}
    
    for key in elem:
        if key == 'lidar':
            # For LiDAR data, we can't stack tensors with different sizes
            # Instead, we'll just keep them as a list of tensors
            result[key] = [d[key] for d in batch]
        elif key == 'calib' or key == 'ego_pose':
            # For dictionaries like calibration data, keep as list
            result[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            # For uniform tensors like images and BEV labels, stack as usual
            result[key] = torch.stack([d[key] for d in batch])
        else:
            # For other data types, keep as list
            result[key] = [d[key] for d in batch]
    
    return result


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
            spatial_size=(128, 128)
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
    def test_dataset(
        self,
        test_loader: DataLoader,
        save_predictions: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Test on a dataset
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
            visualization_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(visualization_dir, exist_ok=True)
        
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to device
            images = batch['image'].to(self.device)
            lidar = batch['lidar']
            
            # Handle variable-sized LiDAR data
            if isinstance(lidar, list):
                # Process each point cloud separately
                points_list = []
                for points in lidar:
                    points_list.append(points.to(self.device))
            else:
                points_list = [lidar.to(self.device)]
            
            # Time the inference
            start_time = time.time()
            
            # Extract features from camera
            image_feats = self.camera_backbone(images)
            
            # Process each point cloud
            lidar_feats_list = []
            for points in points_list:
                # Add batch dimension if missing
                if points.dim() == 2:
                    points = points.unsqueeze(0)
                lidar_feats = self.lidar_backbone(points)
                lidar_feats_list.append(lidar_feats['bev_features']['stage3'])
            
            # Stack LiDAR features if they have same shape
            if all(feat.shape == lidar_feats_list[0].shape for feat in lidar_feats_list):
                lidar_feats = torch.cat(lidar_feats_list, dim=0)
            else:
                # Use the first one as placeholder (this is a simplification)
                lidar_feats = lidar_feats_list[0]
            
            # Ensure lidar_feats has the correct shape [B, C, H, W]
            if lidar_feats.dim() == 5:  # [B, 1, C, H, W]
                lidar_feats = lidar_feats.squeeze(1)
            
            # Create dictionary of lidar features for each stage
            lidar_features_dict = {
                'stage1': lidar_feats,
                'stage2': lidar_feats,
                'stage3': lidar_feats
            }
            
            # Create dictionary of image features for each stage with correct channel dimensions
            # The fusion module expects specific channel dimensions for each stage
            image_features_dict = {}
            
            # Use available feature maps from the backbone
            if 'stage3' in image_feats:
                image_features_dict['stage1'] = image_feats['stage3']  # 256 channels
            elif 'stage5' in image_feats:
                image_features_dict['stage1'] = image_feats['stage5']  # Fallback
            
            if 'stage4' in image_feats:
                image_features_dict['stage2'] = image_feats['stage4']  # 384 channels
            elif 'stage5' in image_feats:
                image_features_dict['stage2'] = image_feats['stage5']  # Fallback
            
            if 'stage5' in image_feats:
                image_features_dict['stage3'] = image_feats['stage5']  # 512 channels
            
            # Fuse features
            fused_feats, _ = self.fusion(lidar_features_dict, image_features_dict)
            
            # Get predictions
            predictions = self.head(fused_feats)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Update metrics if targets are available
            if 'bev_label' in batch:
                targets = batch['bev_label'].to(self.device)
                self.metrics.update(predictions, targets)
            
            # Save predictions if requested
            if save_predictions and output_dir:
                # Get sample tokens if available
                if 'sample_token' in batch:
                    sample_tokens = batch['sample_token']
                else:
                    sample_tokens = [f"sample_{i}_{j}" for j in range(len(predictions))]
                
                # Save each prediction in the batch
                for j, (pred, token) in enumerate(zip(predictions, sample_tokens)):
                    pred_viz_path = os.path.join(visualization_dir, f"{token}_pred.png")
                    self.save_prediction_visualization(pred, pred_viz_path)
                    
                    # Save ground truth if available
                    if 'bev_label' in batch:
                        gt = batch['bev_label'][j]
                        gt_viz_path = os.path.join(visualization_dir, f"{token}_gt.png")
                        self.save_ground_truth_visualization(gt, gt_viz_path)
                    
                    # Save camera image if available
                    if 'image' in batch:
                        img = batch['image'][j].cpu().numpy().transpose(1, 2, 0)
                        img = (img * 255).astype(np.uint8)
                        img_path = os.path.join(visualization_dir, f"{token}_image.png")
                        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Compute metrics
        metrics = self.metrics.get_metrics() if 'bev_label' in batch else {}
        
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
        num_classes = pred.shape[0]
        
        # Create RGB visualization
        vis = np.zeros((pred.shape[1], pred.shape[2], 3), dtype=np.uint8)
        
        # Color code for each class
        colors = [
            (255, 0, 0),    # Red for class 0
            (0, 255, 0),    # Green for class 1
            (0, 0, 255),    # Blue for class 2
            (255, 255, 0),  # Yellow for class 3
            (255, 0, 255),  # Magenta for class 4
            (0, 255, 255)   # Cyan for class 5
        ]
        
        # Add each class prediction with its color
        for i in range(min(num_classes, len(colors))):
            mask = pred[i] > threshold
            if i == 0:  # Background
                continue  # Skip background for better visualization
            vis[mask] = colors[i]
        
        # Save visualization
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    def save_ground_truth_visualization(
        self,
        ground_truth: torch.Tensor,
        save_path: str
    ):
        """
        Save a visualization of the ground truth
        Args:
            ground_truth: [C, H, W] or [H, W] tensor
            save_path: Path to save the visualization
        """
        # Convert to numpy
        gt = ground_truth.cpu().numpy()
        
        # Handle different formats
        if gt.ndim == 3:  # Multiple classes
            num_classes = gt.shape[0]
            vis = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
            colors = [
                (255, 0, 0),    # Red for class 0
                (0, 255, 0),    # Green for class 1
                (0, 0, 255),    # Blue for class 2
                (255, 255, 0),  # Yellow for class 3
                (255, 0, 255),  # Magenta for class 4
                (0, 255, 255)   # Cyan for class 5
            ]
            
            for i in range(min(num_classes, len(colors))):
                mask = gt[i] > 0
                if i == 0:  # Background
                    continue  # Skip background
                vis[mask] = colors[i]
        else:  # Single-channel class map
            # Create a colormap
            cmap = plt.cm.get_cmap('tab10', 256)
            vis = cmap(gt)[..., :3]
            vis = (vis * 255).astype(np.uint8)
        
        # Save visualization
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    @torch.no_grad()
    def inference_single(
        self,
        image: torch.Tensor,
        points: torch.Tensor,
        calib: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Run inference on a single sample
        Args:
            image: [1, C, H, W] image tensor
            points: [1, N, 4] point cloud tensor
            calib: Calibration matrices
        Returns:
            [1, num_classes, H, W] prediction tensor
        """
        # Move inputs to device
        image = image.to(self.device)
        points = points.to(self.device)
        
        # Extract features
        image_feats = self.camera_backbone(image)
        lidar_feats = self.lidar_backbone(points)
        
        # Get BEV features and ensure correct shape
        lidar_bev_feats = lidar_feats['bev_features']['stage3']
        if lidar_bev_feats.dim() == 5:  # [B, 1, C, H, W]
            lidar_bev_feats = lidar_bev_feats.squeeze(1)
        
        # Create dictionary of lidar features for each stage
        lidar_features_dict = {
            'stage1': lidar_bev_feats,
            'stage2': lidar_bev_feats,
            'stage3': lidar_bev_feats
        }
        
        # Create dictionary of image features for each stage with correct channel dimensions
        # The fusion module expects specific channel dimensions for each stage
        image_features_dict = {}
        
        # Use available feature maps from the backbone
        if 'stage3' in image_feats:
            image_features_dict['stage1'] = image_feats['stage3']  # 256 channels
        elif 'stage5' in image_feats:
            image_features_dict['stage1'] = image_feats['stage5']  # Fallback
        
        if 'stage4' in image_feats:
            image_features_dict['stage2'] = image_feats['stage4']  # 384 channels
        elif 'stage5' in image_feats:
            image_features_dict['stage2'] = image_feats['stage5']  # Fallback
        
        if 'stage5' in image_feats:
            image_features_dict['stage3'] = image_feats['stage5']  # 512 channels
        
        # Fuse features
        fused_feats, _ = self.fusion(lidar_features_dict, image_features_dict)
        
        # Get predictions
        predictions = self.head(fused_feats)
        
        return predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Test BEV Fusion model')
    parser.add_argument('--model-config', type=str, required=True, 
                        help='Path to model configuration YAML')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--dataroot', type=str, default='/home/mevi/Documents/bev/nuscenes07',
                        help='Path to NuScenes dataset')
    parser.add_argument('--bev-labels-dir', type=str, default='/home/mevi/Documents/bev/test',
                        help='Path to precomputed BEV labels')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to test on')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Directory to save outputs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization saving')
    parser.add_argument('--no-vis', action='store_false', dest='visualize',
                        help='Disable visualization saving')
    parser.set_defaults(visualize=True)
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Load model configuration
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create configuration for tester
    config = {
        'num_classes': model_config['segmentation_head']['num_classes'],
        'voxel_size': model_config['lidar_backbone']['voxel_size'],
        'point_cloud_range': model_config['lidar_backbone']['point_cloud_range']
    }
    
    # Create dataset
    test_dataset = PrecomputedBEVDataset(
        dataroot=args.dataroot,
        bev_labels_dir=args.bev_labels_dir,
        split=args.split,
        return_tokens=True
    )
    
    print(f"Created dataset with {len(test_dataset)} samples")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    
    # Initialize tester
    tester = Tester(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # Run testing
    print(f"Testing model on {args.split} split...")
    metrics = tester.test_dataset(
        test_loader=test_loader,
        save_predictions=args.visualize,
        output_dir=args.output_dir
    )
    
    # Print metrics
    print("\nTest Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {args.output_dir}")


if __name__ == '__main__':
    main()
