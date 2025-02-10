import os
import numpy as np
import torch
from nuscenes_dataset import NuScenesBEVDataset
from visualize import visualize_bev_and_image, visualize_point_cloud
from mmdet3d.datasets.pipelines import LoadPointsFromFile, LoadImageFromFile


def test_dataset():
    """Test the NuScenesBEVDataset implementation"""
    
    # Define data pipeline
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=4
        ),
        dict(
            type='LoadImageFromFile'
        ),
        # Add more transforms as needed
    ]
    
    # Initialize dataset
    dataset = NuScenesBEVDataset(
        data_root='data/nuscenes',  # Update with your path
        ann_file='data/nuscenes/nuscenes_infos_train_mini.pkl',
        pipeline=pipeline,
        bev_size=(128, 128),
        pc_range=[-25.6, 0, -2, 25.6, 51.2, 4]  # Note y_min=0 for front-only view
    )
    
    print(f"\nDataset initialized with {len(dataset)} samples")
    
    # Test data loading
    for i in range(min(5, len(dataset))):
        print(f"\nProcessing sample {i}...")
        
        try:
            # Get data sample
            data = dataset[i]
            
            # Verify point cloud
            if 'points' in data:
                points = data['points'].numpy()
                print(f"Point cloud shape: {points.shape}")
                print(f"Point cloud range: [{points.min():.2f}, {points.max():.2f}]")
                visualize_point_cloud(points)
            
            # Verify camera image
            if 'img' in data:
                img = data['img']
                print(f"Image shape: {img.shape}")
            
            # Verify map mask
            if 'map_mask' in data:
                map_mask = data['map_mask']
                print(f"Map mask shape: {map_mask.shape}")
                print(f"Unique map classes: {np.unique(map_mask)}")
            
            # Visualize BEV and image
            visualize_bev_and_image(dataset, i, 
                                  save_path=f"test_vis_{i}.png")
            
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        input("\nPress Enter to continue to next sample...")


if __name__ == '__main__':
    test_dataset() 