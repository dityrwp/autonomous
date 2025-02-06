import matplotlib.pyplot as plt
import numpy as np
import cv2
from nuscenes_dataset import NuScenesBEVDataset

def visualize_bev_label(bev_label, save_path=None):
    """
    Visualizes BEV segmentation labels.

    Args:
        bev_label (np.ndarray): BEV segmentation map, shape (2, 128, 128)
                                - Channel 0: Object Class Labels
                                - Channel 1: Attribute Labels (e.g., moving/stationary)
        save_path (str): Path to save the visualization (optional).
    """
    # Extract class labels
    class_mask = bev_label[0]  # First channel (class segmentation)
    attr_mask = bev_label[1]   # Second channel (attribute segmentation, optional)

    # Define colors for different object classes
    class_colors = {
        0: (0, 0, 0),       # Background - Black
        1: (255, 0, 0),     # Car - Red
        2: (0, 255, 0),     # Truck - Green
        3: (0, 0, 255),     # Trailer - Blue
        4: (255, 255, 0),   # Bus - Yellow
        5: (255, 0, 255),   # Construction Vehicle - Magenta
        6: (0, 255, 255),   # Bicycle - Cyan
        7: (128, 128, 128), # Motorcycle - Gray
        8: (255, 165, 0),   # Pedestrian - Orange
        9: (0, 128, 255),   # Traffic Cone - Light Blue
        10: (128, 0, 128)   # Barrier - Purple
    }

    # Convert class mask to RGB image
    bev_visual = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        bev_visual[class_mask == class_id] = color

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot class segmentation
    ax1.imshow(bev_visual)
    ax1.set_title("Class Segmentation")
    ax1.axis("off")

    # Plot attribute segmentation
    attr_visual = ax2.imshow(attr_mask, cmap='tab10')
    ax2.set_title("Attribute Segmentation")
    ax2.axis("off")
    plt.colorbar(attr_visual, ax=ax2, label='Attribute ID')

    plt.tight_layout()

    # Save or show the visualization
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    plt.show()

def visualize_point_cloud(points, title="LiDAR Point Cloud"):
    """
    Visualize LiDAR point cloud from top-down view.
    
    Args:
        points (np.ndarray): Point cloud array of shape (N, 4) containing x, y, z, intensity
        title (str): Plot title
    """
    plt.figure(figsize=(8, 8))
    
    # Create top-down view (x-y plane)
    plt.scatter(points[:, 0], points[:, 1], 
               c=points[:, 2],  # Color by height (z)
               cmap='viridis',
               s=1)  # Point size
    
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.colorbar(label='Height (m)')
    plt.show()

def test_dataset_preprocessing():
    """Test the NuScenes dataset preprocessing pipeline."""
    try:
        # Initialize dataset
        dataset = NuScenesBEVDataset(
            dataroot="C:\\nuscenes_mini",
            split='train',
            sample_ratio=1.0
        )
        
        print(f"Dataset initialized with {len(dataset.samples)} samples")
        
        # Test data loading for first sample
        sample_idx = 0
        print(f"\nProcessing sample {sample_idx}...")
        
        # Get raw data
        sample_data = dataset.get_data_info(sample_idx)
        
        # Visualize point cloud
        points = sample_data['lidar_points']
        print(f"Point cloud shape: {points.shape}")
        visualize_point_cloud(points)
        
        # Visualize BEV segmentation
        bev_label = sample_data['bev_label']
        print(f"BEV label shape: {bev_label.shape}")
        visualize_bev_label(bev_label, save_path="bev_debug.png")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_preprocessing()
