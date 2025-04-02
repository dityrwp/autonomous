#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.precomputed_bev_dataset import PrecomputedBEVDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Test precomputed BEV dataset')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to NuScenes dataset')
    parser.add_argument('--bev-labels-dir', type=str, required=True, help='Path to precomputed BEV labels')
    parser.add_argument('--split', type=str, default='train', help='Dataset split')
    parser.add_argument('--output-dir', type=str, default='output/dataset_test', help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for DataLoader test')
    parser.add_argument('--camera-only', action='store_true', help='Only load camera data')
    parser.add_argument('--lidar-only', action='store_true', help='Only load LiDAR data')
    return parser.parse_args()

def visualize_bev_labels(dataset, indices, output_dir):
    """Visualize BEV labels for specific samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class map from dataset
    class_map = dataset.class_map
    class_names = list(class_map.keys())
    colors = plt.cm.get_cmap('tab10', len(class_names))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Get BEV label
        bev_label = sample['bev_label'].numpy()
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        
        # Plot BEV label
        plt.imshow(bev_label, cmap='tab10', vmin=0, vmax=len(class_names)-1)
        
        # Add colorbar
        cbar = plt.colorbar(ticks=range(len(class_names)))
        cbar.set_ticklabels(class_names)
        
        # Add title
        if 'sample_token' in sample:
            plt.title(f"BEV Label - Sample {idx}\nToken: {sample['sample_token']}")
        else:
            plt.title(f"BEV Label - Sample {idx}")
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"bev_label_{idx}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot camera image if available
        if 'image' in sample:
            plt.figure(figsize=(10, 6))
            
            # Convert to numpy if tensor
            if isinstance(sample['image'], torch.Tensor):
                image = sample['image'].permute(1, 2, 0).numpy()
                # Denormalize if needed
                if image.max() <= 1.0:
                    image = image * 255
            else:
                image = sample['image']
            
            plt.imshow(image.astype(np.uint8))
            plt.title(f"Camera Image - Sample {idx}")
            plt.savefig(os.path.join(output_dir, f"camera_image_{idx}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Visualize LiDAR if available
        if 'lidar' in sample:
            plt.figure(figsize=(10, 10))
            
            # Take top-down view of LiDAR
            lidar = sample['lidar'].numpy()
            
            # Filter points by height for better top-down view
            # In LiDAR coordinates: x is forward, y is left, z is up
            height_min = -1.0  # meters below the sensor
            height_max = 2.0   # meters above the sensor
            
            # Apply height filter
            height_mask = (lidar[:, 2] >= height_min) & (lidar[:, 2] <= height_max)
            filtered_lidar = lidar[height_mask]
            
            # Limit range for better visualization
            x_range = 50  # meters (forward/backward)
            y_range = 50  # meters (left/right)
            range_mask = (filtered_lidar[:, 0] > -x_range) & (filtered_lidar[:, 0] < x_range) & \
                         (filtered_lidar[:, 1] > -y_range) & (filtered_lidar[:, 1] < y_range)
            filtered_lidar = filtered_lidar[range_mask]
            
            if filtered_lidar.shape[0] > 0:
                # Use proper coordinates: x forward, y left
                plt.scatter(filtered_lidar[:, 0], filtered_lidar[:, 1], 
                            c=filtered_lidar[:, 3],  # Use intensity for coloring
                            s=3,  # Larger point size
                            cmap='plasma', 
                            alpha=0.7)
                
                plt.colorbar(label='Intensity')
                plt.title(f"LiDAR Top-Down View - Sample {idx}\nShowing {filtered_lidar.shape[0]} points")
                plt.xlabel('X - Forward (m)')
                plt.ylabel('Y - Left (m)')
                plt.axis('equal')
                
                # Add grid
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Mark ego vehicle position
                plt.plot(0, 0, 'r*', markersize=15, label='Ego Vehicle')
                
                # Draw box showing the BEV grid area
                grid_meters = dataset.grid_size * dataset.resolution
                rect = plt.Rectangle((-grid_meters/2, 0), 
                                     grid_meters, grid_meters,
                                     fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                plt.legend()
            else:
                plt.text(0.5, 0.5, "No LiDAR points within range", 
                        ha='center', va='center', transform=plt.gca().transAxes,
                        fontsize=14)
            
            plt.savefig(os.path.join(output_dir, f"lidar_top_down_{idx}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create overlay of LiDAR points on BEV label
            plt.figure(figsize=(10, 10))
            
            # Plot BEV label
            plt.imshow(bev_label, cmap='tab10', vmin=0, vmax=len(class_names)-1, alpha=0.7)
            
            # Get grid properties
            grid_size = dataset.grid_size
            resolution = dataset.resolution
            
            # Filter LiDAR points by height for overlay
            if filtered_lidar.shape[0] > 0:
                # Convert LiDAR coordinates to grid coordinates
                # BEV grid is centered on ego vehicle with forward being up
                x = (filtered_lidar[:, 1] / resolution + grid_size // 2).astype(int)  # y-coord in LiDAR = left/right in BEV
                y = (-filtered_lidar[:, 0] / resolution + grid_size).astype(int)  # negative x-coord = forward in BEV
                
                # Filter points within grid
                mask = (x >= 0) & (x < grid_size) & (y >= 0) & (y < grid_size)
                x, y = x[mask], y[mask]
                
                if x.size > 0:
                    # Plot LiDAR points
                    plt.scatter(x, y, c=filtered_lidar[mask, 3], s=3, cmap='plasma', alpha=0.7)
                    plt.colorbar(label='Intensity')
                else:
                    plt.text(0.5, 0.1, "No LiDAR points within BEV grid", 
                            ha='center', va='center', transform=plt.gca().transAxes,
                            fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.7))
            else:
                plt.text(0.5, 0.1, "No LiDAR points within range", 
                        ha='center', va='center', transform=plt.gca().transAxes,
                        fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.7))
            
            plt.title(f"LiDAR Points on BEV Label - Sample {idx}")
            plt.savefig(os.path.join(output_dir, f"lidar_bev_overlay_{idx}.png"), dpi=300, bbox_inches='tight')
            plt.close()

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

def test_dataloader(dataset, batch_size):
    """Test DataLoader functionality."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
        collate_fn=custom_collate_fn  # Use our custom collate function
    )
    
    # Time dataloader iteration
    import time
    start_time = time.time()
    
    for i, batch in enumerate(tqdm(loader, desc="Testing DataLoader")):
        # Process first 5 batches
        if i >= 5:
            break
            
        # Print batch information
        print(f"\nBatch {i}:")
        for key, value in batch.items():
            if key == 'lidar':
                # For LiDAR data, show shape of each item in the batch
                print(f"  {key}: List of {len(value)} tensors")
                for j, lidar_tensor in enumerate(value):
                    print(f"    Item {j}: {lidar_tensor.shape}, {lidar_tensor.dtype}")
            elif key == 'calib':
                # For calibration data, show what's included
                print(f"  {key}: List of {len(value)} dictionaries")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}, {value.dtype}")
            elif isinstance(value, list):
                print(f"  {key}: List of {len(value)} items")
            else:
                print(f"  {key}: {type(value)}")
    
    # Calculate loading speed
    end_time = time.time()
    loading_time = end_time - start_time
    samples_per_second = min(5 * batch_size, len(dataset)) / loading_time
    
    print(f"\nDataLoader test completed.")
    print(f"Loading time for {min(5 * batch_size, len(dataset))} samples: {loading_time:.2f}s")
    print(f"Samples per second: {samples_per_second:.2f}")

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = PrecomputedBEVDataset(
        dataroot=args.dataroot,
        bev_labels_dir=args.bev_labels_dir,
        split=args.split,
        camera_only=args.camera_only,
        lidar_only=args.lidar_only
    )
    
    # Print dataset information
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Grid size: {dataset.grid_size}x{dataset.grid_size}")
    print(f"Resolution: {dataset.resolution}m/pixel")
    print(f"Class map: {dataset.class_map}")
    
    # Generate random indices to visualize
    import random
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    
    # Visualize samples
    visualize_bev_labels(dataset, indices, args.output_dir)
    
    # Test DataLoader
    test_dataloader(dataset, args.batch_size)
    
    logging.info(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 